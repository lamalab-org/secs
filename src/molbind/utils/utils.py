from typing import Any, List, Tuple  # noqa: I002, UP035

import torch


def rename_keys_with_prefix(d: dict) -> dict:
    new_dict = {}
    for key, value in d.items():
        if key.startswith("model."):
            # remove the prefix
            new_key = key[len("model.") :]
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


def find_all_pairs_in_list(lst: List[Any]) -> List[Tuple[Any, Any]]:  # noqa: UP006
    """Finds all pairs in a list."""
    return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))]
