from typing import Any

import modal
from main import spec2struct
from pydantic import BaseModel

app = modal.App("spec2struct")
image = (
    modal.Image.from_registry("nvidia/cuda:12.6.2-base-ubuntu22.04", add_python="3.11")
    .add_local_file("./retrieval.py", "/app/retrieval.py", copy=True)
    .add_local_dir("./checkpoints", "/app/checkpoints", copy=True)
    .add_local_dir("./checkpoints_xgb", "/app/checkpoints_xgb", copy=True)
    .add_local_file("./main.py", "/app/main.py", copy=True)
    .add_local_file("./gafuncs.py", "/app/gafuncs.py", copy=True)
    .add_local_file("./isomer_to_canon.py", "/app/isomer_to_canon.py", copy=True)
    .add_local_file("./prune.py", "/app/prune.py", copy=True)
    .add_local_dir("./configs", "/app/configs", copy=True)
    .add_local_file("./filtered_pubchem.parquet", "/app/cache_pubchem.parquet", copy=True)
    .apt_install("git", "wget", "gcc", "g++", "build-essential")
    .run_commands("export POLARS_MAX_THREADS=24")
    .run_commands("which gcc > /tmp/gcc_path")
    .run_commands("ls /app")
    .pip_install("uv")
    .run_commands("uv pip install --system mol_ga")
    .run_commands("uv pip install --system xgboost")
    .run_commands("uv pip install --system scikit-learn")
    .run_commands("uv pip install --system git+https://github.com/lamalab-org/MoleculeBind.git")
    .run_commands("uv pip install --system pydantic")
    .env({"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++", "PYTHONPATH": "/app"})
)


class GenerateRequest(BaseModel):
    mf: str
    spectrum: list[float]


# Alternative version if you want to keep dict input
@app.function(
    image=image,
    gpu="A10g",
    memory=24576,
    timeout=1000,
    max_containers=3,
    cpu=12,
)
@modal.fastapi_endpoint(method="POST")
def elucidate_spectrum(request: dict) -> dict[str, Any]:
    return spec2struct(**request)
