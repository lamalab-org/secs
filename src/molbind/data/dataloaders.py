from lightning.pytorch.utilities.combined_loader import CombinedLoader
from networkx import Graph
from torch.utils.data import DataLoader

from molbind.data.available import MODALITY_DATA_TYPES
from molbind.data.components.datasets import GraphDataset, StringDataset


def load_combined_loader(
    data_modalities: dict,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    central_modality: str = "smiles",
    drop_last: bool = True,
) -> CombinedLoader:
    """Combine multiple dataloaders for different modalities into a single dataloader.

    Args:
        data_modalities (dict): data inputs for each modality as pairs of (central_modality, modality)
        batch_size (int): batch size for the dataloader
        shuffle (bool): shuffle the dataset
        num_workers (int): number of workers for the dataloader
        drop_last (bool, optional): whether to drop the last batch; defaults to True.
        central_modality (str, optional): central modality to use for the dataset; defaults to "smiles".
    Returns:
        CombinedLoader: a combined dataloader for all the modalities
    """
    loaders = {}

    for modality in [*data_modalities]:
        if MODALITY_DATA_TYPES[modality] == str:
            dataset_instance = StringDataset(
                dataset=data_modalities[modality],
                modality=modality,
                central_modality=central_modality,
                context_length=256,
            )
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
