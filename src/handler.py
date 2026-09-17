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

        # SheetSage2 (remote code) before YuE2 — YuE2 custom modules can break
        # transformers lazy imports that SheetSage2 still uses (4.45-style).
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


def transcribe_melody(audio_path: str) -> str:
    _, transcriber = ensure_models()
    log(f"transcribing melody from {audio_path}")
    result = transcriber.transcribe(audio_path, melody_only=True)
    abc = (result or {}).get("abc") if isinstance(result, dict) else None
    if not abc or not str(abc).strip():
        raise RuntimeError("SheetSage2 did not produce melody ABC")
    return str(abc)


def generate_cover(style: str, lyrics: str, abc: str, seed: int, ode_steps: int, cot: str):
    pipe, _ = ensure_models()
    original = pipe.generation_config
    pipe.generation_config = replace(original, ode_steps=int(ode_steps))
    try:
        log(f"generating cover cot={cot} seed={seed} ode_steps={ode_steps}")
        return pipe(style=style, lyrics=lyrics, abc=abc, cot=cot, seed=int(seed))
    finally:
        pipe.generation_config = original


def handler(event):
    inp = event.get("input", {}) or {}
    try:
        if inp.get("warmup"):
            ensure_models()
            return {"ok": True, "warm": True}

        audio_url = inp.get("audio_url")
        abc = (inp.get("abc") or "").strip()
        style = (inp.get("style") or inp.get("style_prompt") or "").strip()
        lyrics = (inp.get("lyrics") or "").strip()
        cot = (inp.get("cot") or "melody").strip()
        seed = int(inp.get("seed", 831001))
        ode_steps = int(inp.get("ode_steps", 16))

        if not style:
            return {"error": "Missing input.style or input.style_prompt"}
        if not lyrics:
            return {"error": "Missing input.lyrics"}
        if cot not in {"melody", "full", "off"}:
            return {"error": "input.cot must be melody, full, or off"}
        if not abc and not audio_url:
            return {"error": "Provide input.audio_url or input.abc"}
        if len(style) > 1000:
            return {"error": "style must be <= 1000 characters"}
        if len(lyrics) > 12000:
            return {"error": "lyrics must be <= 12000 characters"}

        in_audio = None
        if not abc:
            in_audio = download_to_file(audio_url, suffix_from_url(audio_url))
            abc = transcribe_melody(in_audio)

        song = generate_cover(style, lyrics, abc, seed, ode_steps, cot)

        work = Path(tempfile.mkdtemp(prefix="yue2-cover-"))
        flac_path = work / "cover.flac"
        mp3_path = work / "cover.mp3"
        song.save(str(flac_path))
        ffmpeg_to_mp3(str(flac_path), str(mp3_path))
        audio_b64 = base64.b64encode(mp3_path.read_bytes()).decode("ascii")

        return {
            "ok": True,
            "audio_base64": audio_b64,
            "audio_mime": "audio/mpeg",
            "abc": song.abc or abc,
            "seed": seed,
            "cot": cot,
            "ode_steps": ode_steps,
            "sample_rate": song.sample_rate,
            "timing": song.timing,
        }
    except Exception as e:
        log("handler failed:\n" + traceback.format_exc())
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
