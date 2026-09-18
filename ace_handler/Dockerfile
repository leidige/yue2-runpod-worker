# Thin overlay: reuse baked ACE models, only replace handler.
# Build (when Docker is up):
#   docker build -t ghcr.io/leidige/ace-step-handler:srcfix -f ace_handler/Dockerfile .
#   docker push ghcr.io/leidige/ace-step-handler:srcfix
FROM ghcr.io/conalmullan/video-toolkit-acestep:latest
COPY ace_handler/handler.py /app/handler.py
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-u", "/app/handler.py"]
