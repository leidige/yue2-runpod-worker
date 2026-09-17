import base64
import os
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
    abc: str,
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
            f"cfg_scale={cfg_scale}"
        )
        kwargs = {
            "style": style,
            "lyrics": lyrics,
            "abc": abc,
            "cot": cot,
            "seed": int(seed),
        }
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

        # ---- Cover：可用已有谱，或现场转谱后生成 ----
        if not style:
            return {"error": "Missing input.style or input.style_prompt"}
        if not lyrics:
            return {"error": "Missing input.lyrics"}
        if not abc and not audio_url:
            return {"error": "Provide input.audio_url or input.abc"}
        if len(style) > 1000:
            return {"error": "style must be <= 1000 characters"}
        if len(lyrics) > 12000:
            return {"error": "lyrics must be <= 12000 characters"}

        source_abc = abc
        in_audio = None
        try:
            if not abc:
                in_audio = download_to_file(audio_url, suffix_from_url(audio_url))
                source_abc = transcribe_score(in_audio, melody_only=melody_only)
                abc = source_abc

            song = generate_cover(style, lyrics, abc, seed, ode_steps, cot, cfg_scale)

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
