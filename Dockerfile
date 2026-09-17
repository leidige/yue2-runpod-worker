FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# python deps
COPY builder/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# worker code
COPY src/handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
