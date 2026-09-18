#!/usr/bin/env python3
"""
Patched RunPod serverless handler for ACE-Step 1.5.

FIXES vs community image handler:
1) Cover uses src_audio (structure/melody), NOT reference_audio (global timbre only).
2) Turbo-friendly defaults: inference_steps=24, shift=3.0.
3) Optional reference_audio can still be passed separately for timbre.
4) Logs which audio fields were wired (for diagnosis).
"""

from __future__ import annotations

import base64
import os
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import runpod
import torch


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


_dit_handler = None


def setup_hf_cache() -> None:
    baked_cache = Path("/root/.cache/huggingface")
    if baked_cache.exists():
        os.environ["HF_HOME"] = str(baked_cache)
        log(f"Using baked-in model cache: {baked_cache}")
    elif Path("/runpod-volume").exists() and os.access("/runpod-volume", os.W_OK):
        cache_path = Path("/runpod-volume/.cache/huggingface")
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)
        log(f"Using RunPod network volume cache: {cache_path}")


def get_dit_handler():
    global _dit_handler
    if _dit_handler is None:
        log("Loading ACE-Step DiT model...")
        t0 = time.time()
        from acestep.handler import AceStepHandler

        _dit_handler = AceStepHandler()
        config_path = os.environ.get("ACE_FORCE_CONFIG") or os.environ.get(
            "ACESTEP_CONFIG_PATH", "acestep-v15-xl-sft"
        )
        device = os.environ.get("ACESTEP_DEVICE", "cuda")
        log(f"Loading DiT with config_path={config_path}")
        _dit_handler.initialize_service(
            project_root="/app/acestep-repo",
            config_path=config_path,
            device=device,
        )
        log(f"DiT model loaded in {time.time() - t0:.1f}s (config={config_path})")
    return _dit_handler


def _is_turbo(config_path: str) -> bool:
    return "turbo" in (config_path or "").lower()


def _default_steps(config_path: str) -> int:
    # Official: turbo ~8; base/sft commonly 32-64 (XL-SFT table lists 50)
    return 8 if _is_turbo(config_path) else 50


def _default_shift(config_path: str) -> float:
    return 3.0


def _default_guidance(config_path: str) -> float:
    return 1.0 if _is_turbo(config_path) else 7.0


def upload_to_r2(file_path: str, r2_config: dict) -> Optional[str]:
    try:
        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            endpoint_url=r2_config["endpoint_url"],
            aws_access_key_id=r2_config["access_key_id"],
            aws_secret_access_key=r2_config["secret_access_key"],
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        bucket = r2_config["bucket_name"]
        key = f"acestep/{uuid.uuid4().hex}{Path(file_path).suffix}"
        s3.upload_file(file_path, bucket, key)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
        log(f"Uploaded to R2: {key}")
        return url
    except Exception as e:
        log(f"R2 upload failed: {e}")
        return None


def save_temp_audio(audio_base64: str, suffix: str = ".mp3") -> str:
    audio_bytes = base64.b64decode(audio_base64)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def get_audio_duration(file_path: str) -> Optional[float]:
    try:
        import torchaudio

        info = torchaudio.info(file_path)
        return info.num_frames / info.sample_rate
    except Exception:
        pass
    try:
        import subprocess

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def _unlink(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def handler(event: dict) -> dict:
    temp_paths: list[str] = []
    try:
        input_data = event.get("input", {}) or {}
        task_type = str(input_data.get("task_type", "text2music") or "text2music")
        r2_config = input_data.get("r2")
        log(f"[ace-patch] Task={task_type} handler=leidige-src_audio_fix")

        config_path = os.environ.get("ACE_FORCE_CONFIG") or os.environ.get(
            "ACESTEP_CONFIG_PATH", "acestep-v15-xl-sft"
        )
        if "turbo" in config_path.lower() and os.environ.get("ACE_FORCE_CONFIG"):
            return {
                "error": f"refusing turbo config under ACE_FORCE_CONFIG: {config_path}",
                "handler": "leidige-src_audio_fix",
            }
        dit = get_dit_handler()
        t0 = time.time()

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # Re-read after model load (get_dit_handler uses env)
        config_path = os.environ.get("ACE_FORCE_CONFIG") or os.environ.get(
            "ACESTEP_CONFIG_PATH", "acestep-v15-xl-sft"
        )
        prompt = input_data.get("prompt", "") or ""
        lyrics = input_data.get("lyrics", "") or ""
        duration = float(input_data.get("audio_duration", 30))
        steps = int(input_data.get("inference_steps", _default_steps(config_path)))
        audio_format = input_data.get("audio_format", "mp3") or "mp3"
        seed = input_data.get("seed")
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = int(seed)

        shift = float(input_data.get("shift", _default_shift(config_path)))
        guidance_scale = float(input_data.get("guidance_scale", _default_guidance(config_path)))

        params = GenerationParams(
            task_type=task_type,
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            inference_steps=steps,
            seed=seed,
            vocal_language=input_data.get("vocal_language", "unknown"),
            shift=shift,
            guidance_scale=guidance_scale,
            audio_cover_strength=float(input_data.get("audio_cover_strength", 1.0)),
        )

        if input_data.get("bpm"):
            params.bpm = int(input_data["bpm"])
        if input_data.get("key_scale"):
            params.keyscale = str(input_data["key_scale"])
        if input_data.get("time_signature") not in (None, ""):
            params.timesignature = str(input_data["time_signature"])

        # --- Audio wiring (CRITICAL) ---
        # Official Cover: src_audio = structure/melody control
        # reference_audio = global timbre only (does NOT preserve song identity)
        src_b64 = (
            input_data.get("src_audio_base64")
            or (input_data.get("reference_audio_base64") if task_type == "cover" else None)
        )
        ref_b64 = input_data.get("timbre_audio_base64") or input_data.get("style_audio_base64")
        # For cover: uploaded file must go to src_audio. Optional separate timbre ref.
        if task_type == "cover":
            if not src_b64:
                return {
                    "error": "cover requires src_audio_base64 (or reference_audio_base64 mapped to src)",
                    "hint": "ACE Cover structure control uses src_audio, not reference_audio",
                }
            src_path = save_temp_audio(src_b64)
            temp_paths.append(src_path)
            params.src_audio = src_path
            log(f"[ace-patch] cover: src_audio set ({os.path.getsize(src_path)} bytes), strength={params.audio_cover_strength}")
            if ref_b64:
                ref_path = save_temp_audio(ref_b64)
                temp_paths.append(ref_path)
                params.reference_audio = ref_path
                log("[ace-patch] cover: optional reference_audio (timbre) also set")
            else:
                # Do NOT put the same file into reference_audio — that path averages away melody
                log("[ace-patch] cover: reference_audio left empty (structure via src_audio only)")

        elif task_type == "extract":
            src_b64 = input_data.get("src_audio_base64") or input_data.get("reference_audio_base64")
            if not src_b64:
                return {"error": "src_audio_base64 required for extract"}
            src_path = save_temp_audio(src_b64)
            temp_paths.append(src_path)
            params.src_audio = src_path

        elif task_type in ("repainting", "repaint", "lego", "complete", "continuation"):
            src_b64 = input_data.get("src_audio_base64") or input_data.get("reference_audio_base64")
            if src_b64:
                src_path = save_temp_audio(src_b64)
                temp_paths.append(src_path)
                params.src_audio = src_path
            if input_data.get("repainting_start") is not None:
                params.repainting_start = float(input_data["repainting_start"])
            if input_data.get("repainting_end") is not None:
                params.repainting_end = float(input_data["repainting_end"])

        else:
            # text2music: optional global timbre reference only
            timbre = input_data.get("reference_audio_base64") or input_data.get("timbre_audio_base64")
            if timbre:
                ref_path = save_temp_audio(timbre)
                temp_paths.append(ref_path)
                params.reference_audio = ref_path
                log("[ace-patch] text2music: reference_audio (timbre) set")

        config = GenerationConfig(
            batch_size=1,
            audio_format=audio_format,
            seeds=[seed],
            use_random_seed=False,
        )
        save_dir = tempfile.mkdtemp(prefix="acestep_")
        log(
            f"[ace-patch] Generating: task={task_type} dur={duration}s steps={steps} "
            f"shift={shift} seed={seed} src={bool(getattr(params, 'src_audio', None))} "
            f"ref={bool(getattr(params, 'reference_audio', None))}"
        )

        result = generate_music(
            dit_handler=dit,
            llm_handler=None,
            params=params,
            config=config,
            save_dir=save_dir,
        )
        inference_time_ms = int((time.time() - t0) * 1000)

        if not result or not result.success or not result.audios:
            error_msg = result.error if result and result.error else "No audio generated"
            return {"error": str(error_msg), "handler": "leidige-src_audio_fix"}

        audio_entry = result.audios[0]
        output_path = audio_entry.get("path", "")
        audio_params = audio_entry.get("params", {}) or {}
        metas = {
            "bpm": audio_params.get("bpm"),
            "keyscale": audio_params.get("keyscale", audio_params.get("key_scale")),
            "duration": audio_params.get("duration"),
            "timesignature": audio_params.get("timesignature"),
        }
        seed_value = audio_params.get("seed", seed)
        if not output_path or not Path(output_path).exists():
            return {"error": "No output audio file produced", "handler": "leidige-src_audio_fix"}

        actual_duration = get_audio_duration(output_path)
        response = {
            "success": True,
            "seed_value": seed_value,
            "metas": metas,
            "inference_time_ms": inference_time_ms,
            "handler": "leidige-src_audio_fix",
            "wired": {
                "task_type": task_type,
                "config_path": config_path,
                "used_src_audio": bool(getattr(params, "src_audio", None)),
                "used_reference_audio": bool(getattr(params, "reference_audio", None)),
                "audio_cover_strength": float(getattr(params, "audio_cover_strength", 0)),
                "inference_steps": steps,
                "shift": shift,
                "guidance_scale": guidance_scale,
            },
        }
        if actual_duration:
            response["actual_duration_seconds"] = round(actual_duration, 2)

        if r2_config:
            url = upload_to_r2(output_path, r2_config)
            if url:
                response["output_url"] = url
            else:
                with open(output_path, "rb") as f:
                    response["audio_base64"] = base64.b64encode(f.read()).decode("utf-8")
        else:
            with open(output_path, "rb") as f:
                response["audio_base64"] = base64.b64encode(f.read()).decode("utf-8")

        _unlink(output_path)
        log(f"[ace-patch] Done: {inference_time_ms}ms, audio={actual_duration}")
        return response

    except Exception as e:
        log(f"Handler error: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {"error": str(e), "handler": "leidige-src_audio_fix"}
    finally:
        for p in temp_paths:
            _unlink(p)


if __name__ == "__main__":
    setup_hf_cache()
    log("ACE-Step patched handler starting (src_audio cover fix)...")
    log(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    try:
        get_dit_handler()
        log("Model pre-warmed successfully")
    except Exception as e:
        log(f"Pre-warm failed (will retry on first request): {e}")
    runpod.serverless.start({"handler": handler})
