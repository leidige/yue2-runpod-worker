import os, json, tempfile, subprocess, requests
import runpod

# TODO: 你需要在这里接入 YuE2 的实际推理调用
# 目前先把流程打通：下载输入音频 -> 返回原音频(占位)。
# 等你贴 YuE2 仓库的 cover 推理命令/函数入口后，我再把这里替换成真实生成。

def download_to_file(url: str, suffix: str):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return path

def handler(event):
    inp = event.get("input", {}) or {}

    audio_url = inp.get("audio_url")
    style_prompt = inp.get("style_prompt", "")
    lyrics = inp.get("lyrics", "")

    if not audio_url:
        return {"error": "Missing input.audio_url"}

    in_audio = download_to_file(audio_url, suffix=".wav")

    # 验证 ffmpeg/依赖是否存在（防止你上次那种“部署了但跑不起来”）
    try:
        subprocess.check_output(["ffmpeg", "-version"])
    except Exception as e:
        return {"error": f"ffmpeg not available: {e}"}

    # TODO: 在这里调用 YuE2 cover 推理，输出 out_audio
    # out_audio = run_yue2_cover(in_audio, style_prompt, lyrics)
    out_audio = in_audio  # 占位：先回传输入，证明链路OK

    return {
        "ok": True,
        "note": "Pipeline placeholder. Replace with YuE2 cover inference.",
        "inputs_echo": {"style_prompt": style_prompt, "lyrics_len": len(lyrics)},
        "output_audio_path": out_audio,
    }

runpod.serverless.start({"handler": handler})
