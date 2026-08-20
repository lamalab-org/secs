# HOSE 13C shift prediction service.
#
# Built from the repository root so the lookup module can be copied from the
# package rather than duplicated:
#
#     docker build -f services/hose/Dockerfile -t secs/hose .
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
        "rdkit==2025.9.6" \
        "numpy" \
        "fastapi" "uvicorn[standard]"

# The canonical lookup implementation; kept in one place so table and service
# cannot disagree about how environment keys are built.
COPY src/secs/elucidation/verifiers/hose.py /app/hose.py
COPY services/hose/app.py /app/app.py

# The table is data, not code -- mount it at run time:
#   -v $PWD/experiments/hose_table.json:/app/hose_table.json:ro
ENV HOSE_TABLE_PATH=/app/hose_table.json

EXPOSE 7995

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7995/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7995"]
