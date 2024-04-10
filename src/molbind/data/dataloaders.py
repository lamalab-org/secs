from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from molbind.data.components.tokenizers import SMILES_TOKENIZER, SELFIES_TOKENIZER
from networkx import Graph
from typing import Tuple
from torch import Tensor


MODALITY_DATA_TYPES = {
    "smiles": str,
    "selfies": str,
    "graph": Graph,
    "nmr": str,
    "ir": str,
}

STRING_TOKENIZERS = {
    "smiles": SMILES_TOKENIZER,
    "selfies": SELFIES_TOKENIZER,
    "iupac_name": "iupac_name_tokenizer",
}


class StringDataset(Dataset):
    def __init__(
        self, dataset: Tuple[Tensor, Tensor], modality: str, context_length=256
    ):
        """_summary_

        Args:
            dataset (Tuple[Tensor, Tensor]): pair of SMILES and data for the modality (smiles always index 0, modality index 1)
            modality (str): name of data modality as found in MODALITY_DATA_TYPES
            context_length (int, optional): _description_. Defaults to 256.
        """
        assert len(dataset) == 2
        assert len(dataset[0]) == len(dataset[1])
        self.modality = modality
        self.tokenized_smiles = STRING_TOKENIZERS["smiles"](
            dataset[0],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )
        self.tokenized_string = STRING_TOKENIZERS[modality](
            dataset[1],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=context_length,
        )

    def __len__(self):
        return len(self.tokenized_smiles.input_ids)

    def __getitem__(self, idx):
        return {
            "smiles": (
                self.tokenized_smiles.input_ids[idx],
                self.tokenized_smiles.attention_mask[idx],
            ),
            self.modality: (
                self.tokenized_string.input_ids[idx],
                self.tokenized_string.attention_mask[idx],
            ),
        }


class GraphDataset(Dataset):
    pass


def load_combined_loader(
    data_modalities: dict,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    drop_last: bool = True,
) -> CombinedLoader:
    """Combine multiple dataloaders for different modalities into a single dataloader.

    Args:
        data_modalities (dict): data inputs for each modality as pairs of (SMILES, modality)
        batch_size (int): batch size for the dataloader
        shuffle (bool): shuffle the dataset
        num_workers (int): number of workers for the dataloader
        drop_last (bool, optional): whether to drop the last batch; defaults to True.

    Returns:
        CombinedLoader: a combined dataloader for all the modalities
    """
    loaders = {}

    for modality in [*data_modalities]:
        if MODALITY_DATA_TYPES[modality] == str:
            dataset_instance = StringDataset(data_modalities[modality], modality)
            loaders[modality] = DataLoader(
                dataset_instance,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
            )
        elif MODALITY_DATA_TYPES[modality] == Graph:
            graph_dataset_instance = GraphDataset(data_modalities[modality])
            loaders[modality] = DataLoader(
                graph_dataset_instance,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last,
            )
    return CombinedLoader(loaders, mode="sequential")
