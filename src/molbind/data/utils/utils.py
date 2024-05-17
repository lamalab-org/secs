from typing import Tuple  # noqa: UP035, I002

import torch
from torch.utils.data import Dataset, random_split


def split_torch_dataset(
    dataset: Dataset, valid_size: float, seed: int = 0
) -> Tuple[Dataset, Dataset]:  # noqa: UP006
    """Split a torch dataset into train and test sets.

    Args:
        dataset (Dataset): torch dataset
        split_ratio (float): ratio of test set size to dataset size
        seed (int, optional): random seed. Defaults to 0.

    Returns:
        Tuple[Dataset, Dataset]: train and test datasets
    """

    n = len(dataset)
    lengths = [n - int(n * valid_size), int(n * valid_size)]
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))
