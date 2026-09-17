FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128

COPY builder/requirements.txt /app/requirements.txt
COPY builder/yue2_infer-0.1.5-py3-none-any.whl /tmp/yue2_infer-0.1.5-py3-none-any.whl

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir --no-deps /tmp/yue2_infer-0.1.5-py3-none-any.whl \
    && rm -f /tmp/yue2_infer-0.1.5-py3-none-any.whl

COPY src/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
