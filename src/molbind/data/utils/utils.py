from collections import defaultdict  # noqa: I002
from collections.abc import Iterable
from random import Random
from typing import (  # noqa: UP035
    Dict,
    List,
    Tuple,
)

import pandas as pd
import selfies as sf
import torch
from loguru import logger
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset, random_split
from tqdm import tqdm


def canonicalize_smiles(smiles: str) -> str:
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        return None


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


def create_scaffold_split(
    df: pd.DataFrame,
    seed: int,
    frac: List[float],  # noqa: UP006
    entity: str = "SMILES",
):
    """create scaffold split. it first generates molecular scaffold for each molecule
    and then split based on scaffolds
    adapted from: https://github.com/mims-harvard/TDC/tdc/utils/split.py

    Args:
        df (pd.DataFrame): dataset dataframe
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions
        entity (str): the column name for where molecule stores

    Returns:
        dict: a dictionary of splitted dataframes, where keys are train/valid/test
        and values correspond to each dataframe
    """
    return _create_scaffold_split(df[entity], seed, frac)


def smiles_to_selfies(smiles: str) -> str:
    """
    Convert a SELFIES string to a SMILES string.

    Args:
        selfies (str): SELFIES string

    Returns:
        str: SMILES string
    """
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def _create_scaffold_split(
    smiles: Iterable[str],
    seed: int,
    frac: List[float],  # noqa: UP006
) -> Dict[str, pd.DataFrame]:  # noqa: UP006
    """create scaffold split. it first generates molecular scaffold for each molecule
    and then split based on scaffolds
    adapted from: https://github.com/mims-harvard/TDC/tdc/utils/split.py

    Args:
        smiles (Iterable[str]): dataset smiles
        fold_seed (int): the random seed
        frac (list): a list of train/valid/test fractions

    Returns:
        dict: a dictionary of indices for splitted data, where keys are train/valid/test
    """
    random = Random(seed)

    s = smiles
    scaffolds = defaultdict(set)

    error_smiles = 0
    for i, smiles in tqdm(enumerate(s), total=len(s)):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=Chem.MolFromSmiles(smiles), includeChirality=False
            )
            scaffolds[scaffold].add(i)
        except Exception:
            logger.info(smiles + " returns RDKit error and is thus omitted...")
            error_smiles += 1

    train, val, test = [], [], []
    train_size = int((len(s) - error_smiles) * frac[0])
    val_size = int((len(s) - error_smiles) * frac[1])
    test_size = (len(s) - error_smiles) - train_size - val_size
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # index_sets = sorted(list(scaffolds.values()), key=lambda i: len(i), reverse=True)
    index_sets = list(scaffolds.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    if frac[2] == 0:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            else:
                val += index_set
                val_scaffold_count += 1
    else:
        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

    return {
        "train": train,
        "valid": val,
        "test": test,
    }
