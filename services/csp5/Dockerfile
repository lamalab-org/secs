# CSP5 13C shift prediction service.
#
# Kept in its own image to match the cascade/vectordb/forward_synthesis
# service pattern; CSP5 bundles its model weights inside the pip package.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image a fraction of the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir CSP5 "fastapi" "uvicorn[standard]"

COPY app.py /app/app.py

ENV CSP5_BATCH_SIZE=64 \
    CSP5_MAX_EMBED_TRIES=1 \
    CSP5_WORKERS=8 \
    CUDA_VISIBLE_DEVICES=-1

EXPOSE 7996

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7996/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7996"]
