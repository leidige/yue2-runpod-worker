import base64
import os
import re
import subprocess
import tempfile
import threading
import traceback
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import requests
import runpod

DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; yue2-runpod-worker/1.0)",
}
MAX_AUDIO_BYTES = 80 * 1024 * 1024
MODEL_ID = os.environ.get("YUE2_MODEL", "m-a-p/YuE2-3B")
TRANSCRIBER_ID = os.environ.get("SHEETSAGE_MODEL", "m-a-p/SheetSage2")
# Official GenerationConfig default is 32. Cover 默认略抬到 48 换听感余量。
DEFAULT_ODE_STEPS = 48
# 纯伴奏时略抬 CFG，让 Tags（instrumental / not applicable vocal）压过人声习惯
DEFAULT_INSTRUMENTAL_CFG = 1.2
_INSTRUMENTAL_STYLE_RE = re.compile(
    r"instrumental|只要伴奏|纯伴奏|无人声|不要人声|不要唱|no\s*vocals|"
    r"accompaniment\s*only|not applicable vocal",
    re.I,
)
_VOICE_HEADER_RE = re.compile(r"^V:\s*(.+)$", re.I)
_NOTE_LINE_RE = re.compile(r"[A-Ga-g]")
# ABC quoted chord symbols e.g. "Am7" "G/B" — remove for cot=melody covers
_CHORD_QUOTE_RE = re.compile(
    r'"\s*[A-G](?:#|b)?'
    r'(?:maj|min|dim|aug|sus|add|m|M)?'
    r'(?:[0-9]+)?(?:/[A-G](?:#|b)?)?\s*"'
)
_HANDLER_BUILD = "cover-6-official-dual-voice"


_lock = threading.Lock()
_pipe = None
_transcriber = None


def log(msg: str) -> None:
    print(msg, flush=True)


def download_to_file(url: str, suffix: str) -> str:
    with requests.get(url, stream=True, timeout=180, headers=DOWNLOAD_HEADERS) as r:
        r.raise_for_status()
        fd, path = tempfile.mkstemp(suffix=suffix)
        written = 0
        with os.fdopen(fd, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_AUDIO_BYTES:
                    raise ValueError(f"audio_url exceeds {MAX_AUDIO_BYTES} bytes")
                f.write(chunk)
    return path


def suffix_from_url(url: str) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}:
        return ext
    return ".wav"


def ffmpeg_to_mp3(src: str, dst: str) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src, "-codec:a", "libmp3lame", "-b:a", "192k", dst,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def ffmpeg_to_wav(src: str, dst: str) -> None:
    """16-bit PCM WAV，便于下载；比 float WAV 更小。"""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src, "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", dst,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def probe_audio_seconds(path: str) -> float | None:
    """用 ffprobe 读参考成曲时长（秒）。失败返回 None。"""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        sec = float((out.stdout or "").strip())
        if sec > 0 and sec < 3600:
            return sec
    except Exception as exc:
        log(f"ffprobe duration failed: {exc}")
    return None


# YuE2 semantic ≈ 25 codec frames/s（社区/MLX 文档常用换算）
SEMANTIC_FPS = 25.0
# 上下文上限 24576；给前缀（含 ABC）留余量
MAX_SEMANTIC_TOKENS = 12000


def semantic_bounds_for_duration(seconds: float) -> tuple[int, int]:
    """按参考时长估算 semantic min/max_tokens，减轻「4 分半变 1 分半」。"""
    sec = max(30.0, min(float(seconds), 8 * 60.0))
    target = int(round(sec * SEMANTIC_FPS))
    # 最低生成约 90% 目标时长，避免过早 MUSIC_END
    min_tokens = max(800, int(target * 0.90))
    max_tokens = min(MAX_SEMANTIC_TOKENS, max(min_tokens + 200, int(target * 1.12) + 100))
    if min_tokens >= max_tokens:
        min_tokens = max(200, max_tokens - 200)
    return min_tokens, max_tokens


def expand_instrumental_lyrics_for_duration(seconds: float | None) -> str:
    """空词结构段太少时模型会早停；按时长多铺几段空结构。"""
    sec = float(seconds or 180.0)
    # 约每 40s 一段结构骨架
    n = max(4, min(12, int(round(sec / 40.0))))
    labels = ["verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
    blocks: list[str] = []
    for i in range(n):
        tag = labels[i % len(labels)]
        blocks.append(f"[{tag}]\n\n\n\n\n")
    return "".join(blocks)


def _compat_transformers_exports():
    """SheetSage2 imports PreTrainedModel via lazy transformers; warm real symbols first."""
    import transformers
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.auto import AutoModel
    from transformers.models.bart.configuration_bart import BartConfig

    setattr(transformers, "PreTrainedModel", PreTrainedModel)
    setattr(transformers, "BartConfig", BartConfig)
    setattr(transformers, "AutoModel", AutoModel)
    return AutoModel


def ensure_models():
    global _pipe, _transcriber
    with _lock:
        if _pipe is not None and _transcriber is not None:
            return _pipe, _transcriber

        import torch
        from yue2 import YuE2Pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this worker")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16")

        AutoModel = _compat_transformers_exports()
        log(f"loading SheetSage2 {TRANSCRIBER_ID}")
        _transcriber = AutoModel.from_pretrained(
            TRANSCRIBER_ID,
            trust_remote_code=True,
        ).eval().to("cuda")

        log(f"loading YuE2 pipeline {MODEL_ID}")
        _pipe = YuE2Pipeline.from_pretrained(
            MODEL_ID,
            device="cuda",
            backend="torch",
            progress=True,
        )
        log("models ready")
        return _pipe, _transcriber


def wants_instrumental(style: str, lyrics: str = "", flag: object = None) -> bool:
    if flag is True or str(flag).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    blob = f"{style or ''}\n{lyrics or ''}"
    return bool(_INSTRUMENTAL_STYLE_RE.search(blob))


def _voice_is_vocal(header_body: str) -> bool:
    """SheetSage2 ABC 常见 V:Vocal / V:1 name=\"Vocal\"。"""
    body = (header_body or "").strip()
    low = body.lower()
    if re.search(r"\bname\s*=\s*\"?vocal\"?", low):
        return True
    token = re.split(r"[\s\"'=]+", body, maxsplit=1)[0].strip().lower()
    return token in {"vocal", "vocals", "voice", "singer", "vox"}


def _count_note_lines(lines: list[str]) -> int:
    return sum(
        1
        for ln in lines
        if _NOTE_LINE_RE.search(ln) and not ln.strip().startswith("%") and not ln.strip().startswith("w:")
    )


def strip_chord_symbols_from_abc(abc: str) -> tuple[str, dict]:
    """去掉 ABC 引号和弦标注，避免 cot=melody 时把和声符号当旋律条件。"""
    raw = abc or ""
    cleaned, n = _CHORD_QUOTE_RE.subn("", raw)
    # 压缩因删和弦留下的多余空格（保留换行）
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned, {"chord_quotes_removed": int(n)}


def strip_vocal_voices_from_abc(abc: str) -> tuple[str, dict]:
    """纯伴奏 Cover：去掉 Vocal 声部，只留 Ins/器乐旋律，避免模型跟着人声线哼唱。

    SheetSage2 melody_only=True 会同时保留 Vocal + Ins。YuE2 锁 ABC 后人声标签再强
    也容易把 Vocal 旋律唱出来（issue #18/#127 + Cover 机制）。
    """
    raw = (abc or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return raw, {"stripped": False, "reason": "empty"}

    lines = raw.split("\n")
    out: list[str] = []
    skipping = False
    saw_voice = False
    dropped_vocal = False
    kept_ins = False

    for line in lines:
        m = _VOICE_HEADER_RE.match(line.strip())
        if m:
            saw_voice = True
            if _voice_is_vocal(m.group(1)):
                skipping = True
                dropped_vocal = True
                continue
            skipping = False
            kept_ins = True
            out.append(line)
            continue
        if skipping:
            continue
        out.append(line)

    text = "\n".join(out).strip() + ("\n" if raw.endswith("\n") else "")
    note_lines = _count_note_lines(out)
    meta = {
        "stripped": dropped_vocal,
        "saw_voice_headers": saw_voice,
        "kept_ins_voice": kept_ins,
        "note_lines_after": note_lines,
    }
    if dropped_vocal and note_lines < 2:
        # 剥离后几乎没音高：回退原谱，避免空谱崩生成
        meta["stripped"] = False
        meta["reason"] = "fallback_too_few_notes"
        return abc, meta
    if not dropped_vocal:
        meta["reason"] = "no_vocal_voice_header"
    return text, meta


def prefer_vocal_melody_abc(abc: str) -> tuple[str, dict]:
    """人声 Cover：若谱里同时有 Vocal + Ins，只保留 Vocal 主旋律线。

    官方建议 Cover 先选定人声/主旋律再渲染；双声部一起锁容易漂。
    """
    raw = (abc or "").replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return raw, {"preferred": False, "reason": "empty"}

    lines = raw.split("\n")
    has_vocal = False
    has_other = False
    for line in lines:
        m = _VOICE_HEADER_RE.match(line.strip())
        if not m:
            continue
        if _voice_is_vocal(m.group(1)):
            has_vocal = True
        else:
            has_other = True

    if not (has_vocal and has_other):
        return abc, {
            "preferred": False,
            "reason": "single_or_no_voice",
            "has_vocal": has_vocal,
            "has_other": has_other,
            "note_lines_after": _count_note_lines(lines),
        }

    out: list[str] = []
    skipping = False
    for line in lines:
        m = _VOICE_HEADER_RE.match(line.strip())
        if m:
            if _voice_is_vocal(m.group(1)):
                skipping = False
                out.append(line)
            else:
                skipping = True
            continue
        if skipping:
            continue
        out.append(line)

    note_lines = _count_note_lines(out)
    meta = {
        "preferred": True,
        "kept": "vocal",
        "dropped_other_voices": True,
        "note_lines_after": note_lines,
    }
    if note_lines < 4:
        meta["preferred"] = False
        meta["reason"] = "fallback_too_few_notes"
        return abc, meta
    text = "\n".join(out).strip() + ("\n" if raw.endswith("\n") else "")
    return text, meta


def prepare_cover_abc(
    abc: str | None,
    *,
    cot: str,
    instrumental: bool,
    prefer_vocal_only: bool = False,
) -> tuple[str | None, dict]:
    """Cover 谱预处理。

    官方 Cover：melody_only 保留 Vocal+Ins、去掉和弦。默认不要剥 Ins。
    仅纯伴奏才剥 Vocal；prefer_vocal_only 为实验开关。
    """
    info: dict = {
        "handler_build": _HANDLER_BUILD,
        "cot": cot,
        "instrumental": instrumental,
        "prefer_vocal_only": prefer_vocal_only,
    }
    if not abc or not str(abc).strip():
        info["skipped"] = "empty_abc"
        return abc, info

    text = str(abc)
    text, chord_meta = strip_chord_symbols_from_abc(text)
    info["chords"] = chord_meta

    if instrumental:
        text, strip_meta = strip_vocal_voices_from_abc(text)
        info["abc_strip"] = strip_meta
    elif prefer_vocal_only and cot == "melody":
        text, pref_meta = prefer_vocal_melody_abc(text)
        info["vocal_prefer"] = pref_meta
    else:
        info["voices"] = "keep_vocal_and_ins"

    info["note_lines"] = _count_note_lines(text.split("\n"))
    info["abc_chars"] = len(text)
    log(f"cover abc prep={info}")
    return text, info


def prepare_instrumental_cover(
    *,
    style: str,
    lyrics: str,
    abc: str | None,
    cfg_scale: float | None,
) -> tuple[str | None, float | None, dict]:
    """纯伴奏：剥 Vocal ABC + 默认抬 cfg。"""
    info: dict = {"instrumental": True}
    abc_out = abc
    if abc and str(abc).strip():
        abc_out, strip_meta = strip_vocal_voices_from_abc(str(abc))
        info["abc_strip"] = strip_meta
        log(f"instrumental abc_strip={strip_meta}")
    cfg_out = cfg_scale
    if cfg_out is None:
        cfg_out = DEFAULT_INSTRUMENTAL_CFG
        info["cfg_defaulted"] = DEFAULT_INSTRUMENTAL_CFG
        log(f"instrumental: cfg_scale default -> {DEFAULT_INSTRUMENTAL_CFG}")
    return abc_out, cfg_out, info


def preprocess_audio_for_transcribe(src: str) -> str:
    """转谱前规范化：定响度 + 44.1k 立体声 WAV，降低 SheetSage 抓飘概率。"""
    fd, dst = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", src,
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", "44100", "-ac", "2",
                dst,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        log(f"preprocess wav ready bytes={os.path.getsize(dst)}")
        return dst
    except Exception as exc:
        log(f"preprocess failed, use original: {exc}")
        try:
            os.unlink(dst)
        except OSError:
            pass
        return src


def transcribe_score(audio_path: str, *, melody_only: bool) -> tuple[str, dict]:
    _, transcriber = ensure_models()
    clean = preprocess_audio_for_transcribe(audio_path)
    cleanup_clean = clean != audio_path
    try:
        log(f"transcribing score melody_only={melody_only} from {clean}")
        result = transcriber.transcribe(clean, melody_only=bool(melody_only))
        if not isinstance(result, dict):
            raise RuntimeError("SheetSage2 returned non-dict result")
        abc = result.get("abc")
        if result.get("abc_error"):
            raise RuntimeError(f"SheetSage2 abc_error: {result.get('abc_error')}")
        if not abc or not str(abc).strip():
            raise RuntimeError("SheetSage2 did not produce ABC score")
        meta = {
            "warnings": result.get("warnings") or [],
            "melody_only": bool(melody_only),
            "preprocessed": cleanup_clean,
            "abc_chars_raw": len(str(abc)),
        }
        return str(abc), meta
    finally:
        if cleanup_clean:
            try:
                os.unlink(clean)
            except OSError:
                pass


def generate_cover(
    style: str,
    lyrics: str,
    abc: str | None,
    seed: int,
    ode_steps: int,
    cot: str,
    cfg_scale: float | None,
    *,
    target_seconds: float | None = None,
):
    pipe, _ = ensure_models()
    original = pipe.generation_config
    pipe.generation_config = replace(original, ode_steps=int(ode_steps))
    try:
        kwargs = {
            "style": style,
            "lyrics": lyrics,
            "cot": cot,
            "seed": int(seed),
        }
        if abc and str(abc).strip():
            kwargs["abc"] = str(abc).strip()
        if cfg_scale is not None:
            kwargs["cfg_scale"] = float(cfg_scale)
        if target_seconds and target_seconds > 0:
            from yue2.protocol import Sampling

            min_tok, max_tok = semantic_bounds_for_duration(target_seconds)
            kwargs["semantic_sampling"] = Sampling(min_tokens=min_tok, max_tokens=max_tok)
            log(
                f"duration target={target_seconds:.1f}s -> semantic "
                f"min_tokens={min_tok} max_tokens={max_tok}"
            )
        log(
            f"generating cover cot={cot} seed={seed} ode_steps={ode_steps} "
            f"cfg_scale={cfg_scale} has_abc={bool(abc and str(abc).strip())}"
        )
        return pipe(**kwargs)
    finally:
        pipe.generation_config = original


def handler(event):
    inp = event.get("input", {}) or {}
    try:
        if inp.get("warmup"):
            ensure_models()
            return {"ok": True, "warm": True, "handler_build": _HANDLER_BUILD}

        action = (inp.get("action") or "cover").strip().lower()
        if action not in {"cover", "transcribe"}:
            return {"error": "input.action must be cover or transcribe"}

        audio_url = inp.get("audio_url")
        abc = (inp.get("abc") or "").strip()
        style = (inp.get("style") or inp.get("style_prompt") or "").strip()
        lyrics = (inp.get("lyrics") or "").strip()
        cot = (inp.get("cot") or "melody").strip()
        seed = int(inp.get("seed", 831001))
        ode_steps = int(inp.get("ode_steps", DEFAULT_ODE_STEPS))
        melody_only = bool(inp.get("melody_only", True if cot == "melody" else False))
        cfg_raw = inp.get("cfg_scale", None)
        cfg_scale = None if cfg_raw is None or cfg_raw == "" else float(cfg_raw)

        if cot not in {"melody", "full", "off"}:
            return {"error": "input.cot must be melody, full, or off"}
        if ode_steps < 1 or ode_steps > 128:
            return {"error": "ode_steps must be in [1, 128]"}
        if cfg_scale is not None and not (0 <= cfg_scale <= 20):
            return {"error": "cfg_scale must be in [0, 20]"}

        # ---- 仅转谱：先出旋律/和弦谱，供前端审阅修改 ----
        if action == "transcribe":
            if not audio_url:
                return {"error": "transcribe requires input.audio_url"}
            in_audio = download_to_file(audio_url, suffix_from_url(audio_url))
            try:
                source_abc, tx_meta = transcribe_score(in_audio, melody_only=melody_only)
                prepared, prep_meta = prepare_cover_abc(
                    source_abc,
                    cot="melody" if melody_only else "full",
                    instrumental=False,
                    prefer_vocal_only=bool(inp.get("prefer_vocal_only")),
                )
            finally:
                try:
                    os.unlink(in_audio)
                except OSError:
                    pass
            return {
                "ok": True,
                "action": "transcribe",
                "abc": prepared or source_abc,
                "source_abc": source_abc,
                "melody_only": melody_only,
                "transcribe_meta": tx_meta,
                "cover_prep": prep_meta,
                "handler_build": _HANDLER_BUILD,
            }

        # ---- 生成：可 Cover（音频/谱），也可纯风格+歌词文生曲 ----
        if not style:
            return {"error": "Missing input.style or input.style_prompt"}
        if not lyrics:
            return {"error": "Missing input.lyrics"}
        if len(style) > 1000:
            return {"error": "style must be <= 1000 characters"}
        if len(lyrics) > 12000:
            return {"error": "lyrics must be <= 12000 characters"}

        # Cover 默认强制 melody + melody_only，避免误用 off/full 漂旋律
        has_ref = bool(abc) or bool(audio_url)
        if has_ref and cot not in {"melody", "full"}:
            log(f"cover: cot={cot} -> melody")
            cot = "melody"
        if has_ref and cot == "melody":
            melody_only = True

        source_abc = abc
        in_audio = None
        source_seconds: float | None = None
        transcribe_meta: dict = {}
        cover_prep: dict = {}
        try:
            if not abc and audio_url:
                in_audio = download_to_file(audio_url, suffix_from_url(audio_url))
                source_seconds = probe_audio_seconds(in_audio)
                if source_seconds:
                    log(f"reference audio duration={source_seconds:.1f}s")
                source_abc, transcribe_meta = transcribe_score(
                    in_audio, melody_only=melody_only
                )
                abc = source_abc
            elif not abc and not audio_url:
                # 文生曲：不锁参考旋律；melody 无谱时改用 full 让模型自己规划
                if cot == "melody":
                    cot = "full"
                    log("text2music: no abc/audio, cot melody→full")
                source_abc = ""
                abc = ""

            # 可选：调用方直接传目标秒数（无本地文件时）
            if source_seconds is None:
                raw_sec = inp.get("target_seconds") or inp.get("source_seconds")
                if raw_sec not in (None, ""):
                    try:
                        source_seconds = float(raw_sec)
                    except (TypeError, ValueError):
                        source_seconds = None

            instrumental = wants_instrumental(
                style, lyrics, inp.get("force_instrumental") or inp.get("instrumental")
            )
            instrumental_meta: dict = {"instrumental": instrumental}

            if abc and str(abc).strip():
                abc, cover_prep = prepare_cover_abc(
                    abc,
                    cot=cot,
                    instrumental=instrumental,
                    prefer_vocal_only=bool(inp.get("prefer_vocal_only")),
                )

            if instrumental:
                # prepare_cover_abc 已剥 Vocal；这里只补 cfg / 歌词结构
                _, cfg_scale, instrumental_meta = prepare_instrumental_cover(
                    style=style,
                    lyrics=lyrics,
                    abc=None,
                    cfg_scale=cfg_scale,
                )
                if cover_prep:
                    instrumental_meta["cover_prep"] = cover_prep
                # 空词段太少 → 时长塌缩；按时长加铺结构
                lyrics = expand_instrumental_lyrics_for_duration(source_seconds)
                instrumental_meta["lyrics_expanded_for_duration"] = True
                instrumental_meta["lyrics_chars"] = len(lyrics)
                log(f"instrumental cover prepared meta={instrumental_meta}")

            song = generate_cover(
                style,
                lyrics,
                abc if abc else None,
                seed,
                ode_steps,
                cot,
                cfg_scale,
                target_seconds=source_seconds,
            )

            work = Path(tempfile.mkdtemp(prefix="yue2-cover-"))
            flac_path = work / "cover.flac"
            wav_path = work / "cover.wav"
            mp3_path = work / "cover.mp3"
            song.save(str(flac_path))
            ffmpeg_to_wav(str(flac_path), str(wav_path))
            ffmpeg_to_mp3(str(flac_path), str(mp3_path))
            mp3_bytes = mp3_path.read_bytes()
            wav_bytes = wav_path.read_bytes()
            flac_bytes = flac_path.read_bytes()
            log(
                f"exports mp3={len(mp3_bytes)} wav={len(wav_bytes)} flac={len(flac_bytes)} bytes"
            )

            result_abc = getattr(song, "abc", None) or abc
            out_seconds = None
            try:
                out_seconds = float(len(song.audio)) / float(song.sample_rate)
            except Exception:
                out_seconds = probe_audio_seconds(str(mp3_path))
            trunc = getattr(song, "truncated", None)
            return {
                "ok": True,
                "action": "cover",
                "handler_build": _HANDLER_BUILD,
                # 试听默认 MP3；无损另附 WAV / FLAC
                "audio_base64": base64.b64encode(mp3_bytes).decode("ascii"),
                "audio_mime": "audio/mpeg",
                "audio_wav_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "audio_wav_mime": "audio/wav",
                "audio_flac_base64": base64.b64encode(flac_bytes).decode("ascii"),
                "audio_flac_mime": "audio/flac",
                "audio_bytes": {
                    "mp3": len(mp3_bytes),
                    "wav": len(wav_bytes),
                    "flac": len(flac_bytes),
                },
                "abc": result_abc,
                "source_abc": source_abc,
                "seed": seed,
                "cot": cot,
                "ode_steps": ode_steps,
                "cfg_scale": cfg_scale,
                "melody_only": melody_only,
                "instrumental": instrumental,
                "instrumental_meta": instrumental_meta,
                "transcribe_meta": transcribe_meta,
                "cover_prep": cover_prep,
                "source_seconds": source_seconds,
                "output_seconds": out_seconds,
                "truncated": trunc,
                "sample_rate": song.sample_rate,
                "timing": song.timing,
            }
        finally:
            if in_audio:
                try:
                    os.unlink(in_audio)
                except OSError:
                    pass
    except Exception as e:
        log("handler failed:\n" + traceback.format_exc())
        return {"error": str(e), "handler_build": _HANDLER_BUILD}

runpod.serverless.start({"handler": handler})
