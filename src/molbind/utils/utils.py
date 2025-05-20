from typing import Any

import pandas as pd
import torch

HANDLERS = {
    ".csv": pd.read_csv,
    ".pickle": pd.read_pickle,
    ".pkl": pd.read_pickle,
    ".parquet": pd.read_parquet,
}


def rename_keys_with_prefix(d: dict, prefix: str = "model.") -> dict:
    new_dict = {}
    for key, value in d.items():
        if key.startswith(prefix):
            # remove the prefix
            new_key = key[len(prefix) :]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict


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
