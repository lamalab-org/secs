import os
import re
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from loguru import logger

HANDLERS = {
    ".csv": pd.read_csv,
    ".pickle": pd.read_pickle,
    ".pkl": pd.read_pickle,
    ".parquet": pd.read_parquet,
    "": lambda x: load_dataset(x).to_pandas(),
}


def select_device() -> str:
    """Selects the device to use for the model."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def find_all_pairs_in_list(lst: list[Any]) -> list[tuple[Any, Any]]:
    """Finds all pairs in a list."""
    return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))]


def _get_first_node():
    """Return the first node we can find in the Slurm node list."""
    nodelist = os.getenv("SLURM_JOB_NODELIST")

    bracket_re = re.compile(r"(.*?)\[(.*?)\]")
    dash_re = re.compile("(.*?)-")
    comma_re = re.compile("(.*?),")

    bracket_result = bracket_re.match(nodelist)

    if bracket_result:
        node = bracket_result[1]
        indices = bracket_result[2]

        comma_result = comma_re.match(indices)
        if comma_result:
            indices = comma_result[1]

        dash_result = dash_re.match(indices)
        first_index = dash_result[1] if dash_result else indices

        return node + first_index

    comma_result = comma_re.match(nodelist)
    if comma_result:
        return comma_result[1]

    return nodelist


def init_distributed_mode(port: int = 12354):
    """Initialize some environment variables for PyTorch Distributed
    using Slurm.
    """
    # The number of total processes started by Slurm.
    os.environ["WORLD_SIZE"] = os.getenv("SLURM_NTASKS")
    # Index of the current process.
    os.environ["RANK"] = os.getenv("SLURM_PROCID")
    # Index of the current process on this node only.
    os.environ["LOCAL_RANK"] = os.getenv("SLURM_LOCALID")

    master_addr = _get_first_node()
    systemname = os.getenv("SYSTEMNAME", "")
    # Need to append "i" on Jülich machines to connect across InfiniBand cells.
    if systemname in ["juwels", "juwelsbooster", "jureca"]:
        master_addr = master_addr + "i"
    os.environ["MASTER_ADDR"] = master_addr

    # An arbitrary free port on node 0.
    os.environ["MASTER_PORT"] = str(port)
    # print the environment variables
    logger.info(f"MASTER_ADDR={os.getenv('MASTER_ADDR')}")
    logger.info(f"MASTER_PORT={os.getenv('MASTER_PORT')}")
    logger.info(f"WORLD_SIZE={os.getenv('WORLD_SIZE')}")
    logger.info(f"RANK={os.getenv('RANK')}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")
