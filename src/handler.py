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
# Official GenerationConfig default is 32; older Cover path used 16 (half steps).
DEFAULT_ODE_STEPS = 32
# 纯伴奏时略抬 CFG，让 Tags（instrumental / not applicable vocal）压过人声习惯
DEFAULT_INSTRUMENTAL_CFG = 1.2
_INSTRUMENTAL_STYLE_RE = re.compile(
    r"instrumental|只要伴奏|纯伴奏|无人声|不要人声|不要唱|no\s*vocals|"
    r"accompaniment\s*only|not applicable vocal",
    re.I,
)
_VOICE_HEADER_RE = re.compile(r"^V:\s*(.+)$", re.I)
_NOTE_LINE_RE = re.compile(r"[A-Ga-g]")

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
    note_lines = sum(1 for ln in out if _NOTE_LINE_RE.search(ln) and not ln.strip().startswith("%"))
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


def transcribe_score(audio_path: str, *, melody_only: bool) -> str:
    _, transcriber = ensure_models()
    log(f"transcribing score melody_only={melody_only} from {audio_path}")
    result = transcriber.transcribe(audio_path, melody_only=bool(melody_only))
    abc = (result or {}).get("abc") if isinstance(result, dict) else None
    if not abc or not str(abc).strip():
        raise RuntimeError("SheetSage2 did not produce ABC score")
    return str(abc)


def generate_cover(
    style: str,
    lyrics: str,
    abc: str | None,
    seed: int,
    ode_steps: int,
    cot: str,
    cfg_scale: float | None,
):
    pipe, _ = ensure_models()
    original = pipe.generation_config
    pipe.generation_config = replace(original, ode_steps=int(ode_steps))
    try:
        log(
            f"generating cover cot={cot} seed={seed} ode_steps={ode_steps} "
            f"cfg_scale={cfg_scale} has_abc={bool(abc and str(abc).strip())}"
        )
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
        return pipe(**kwargs)
    finally:
        pipe.generation_config = original


def handler(event):
    inp = event.get("input", {}) or {}
    try:
        if inp.get("warmup"):
            ensure_models()
            return {"ok": True, "warm": True}

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
                source_abc = transcribe_score(in_audio, melody_only=melody_only)
            finally:
                try:
                    os.unlink(in_audio)
                except OSError:
                    pass
            return {
                "ok": True,
                "action": "transcribe",
                "abc": source_abc,
                "source_abc": source_abc,
                "melody_only": melody_only,
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

        source_abc = abc
        in_audio = None
        try:
            if not abc and audio_url:
                in_audio = download_to_file(audio_url, suffix_from_url(audio_url))
                source_abc = transcribe_score(in_audio, melody_only=melody_only)
                abc = source_abc
            elif not abc and not audio_url:
                # 文生曲：不锁参考旋律；melody 无谱时改用 full 让模型自己规划
                if cot == "melody":
                    cot = "full"
                    log("text2music: no abc/audio, cot melody→full")
                source_abc = ""
                abc = ""

            instrumental = wants_instrumental(
                style, lyrics, inp.get("force_instrumental") or inp.get("instrumental")
            )
            instrumental_meta: dict = {"instrumental": instrumental}
            if instrumental:
                abc, cfg_scale, instrumental_meta = prepare_instrumental_cover(
                    style=style,
                    lyrics=lyrics,
                    abc=abc if abc else None,
                    cfg_scale=cfg_scale,
                )
                log(f"instrumental cover prepared meta={instrumental_meta}")

            song = generate_cover(
                style,
                lyrics,
                abc if abc else None,
                seed,
                ode_steps,
                cot,
                cfg_scale,
            )

            work = Path(tempfile.mkdtemp(prefix="yue2-cover-"))
            flac_path = work / "cover.flac"
            mp3_path = work / "cover.mp3"
            song.save(str(flac_path))
            ffmpeg_to_mp3(str(flac_path), str(mp3_path))
            audio_b64 = base64.b64encode(mp3_path.read_bytes()).decode("ascii")

            result_abc = getattr(song, "abc", None) or abc
            return {
                "ok": True,
                "action": "cover",
                "audio_base64": audio_b64,
                "audio_mime": "audio/mpeg",
                "abc": result_abc,
                "source_abc": source_abc,
                "seed": seed,
                "cot": cot,
                "ode_steps": ode_steps,
                "cfg_scale": cfg_scale,
                "melody_only": melody_only,
                "instrumental": instrumental,
                "instrumental_meta": instrumental_meta,
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
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
