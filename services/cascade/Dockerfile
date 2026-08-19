# CASCADE-2.0 13C shift prediction service.
#
# Kept in its own image because CASCADE pins Python 3.10 + TensorFlow 2.11,
# which cannot coexist with the SECS package (Python 3.11-3.12 + PyTorch).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the versions in the upstream cascadeV2.yml. `nfp` is vendored
# inside the model directory, so it is deliberately not installed here.
RUN pip install --no-cache-dir \
        "tensorflow==2.11.0" \
        "tensorflow-probability==0.19.0" \
        "tensorflow-addons==0.23.0" \
        "kgcnn==2.2.1" \
        "rdkit==2023.9.5" \
        "numpy==1.26.4" \
        "pandas==1.5.3" \
        "ase==3.25.0" \
        "fastapi" "uvicorn[standard]"

# Only the SMILES/GPR model is needed; a full clone is ~308 MB.
ARG CASCADE_REF=main
RUN git clone --filter=blob:none --sparse --depth 1 -b ${CASCADE_REF} \
        https://github.com/asbhd/CASCADE-2.0.git /tmp/cascade \
    && cd /tmp/cascade \
    && git sparse-checkout set models/Predict_SMILES_FF_GPR \
    && mv models/Predict_SMILES_FF_GPR /app/model \
    && find /app/model -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} + \
    && rm -rf /tmp/cascade

COPY app.py /app/app.py

ENV CASCADE_MODEL_DIR=/app/model \
    CASCADE_BATCH_SIZE=32 \
    CUDA_VISIBLE_DEVICES=-1

EXPOSE 7997

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7997/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7997"]
