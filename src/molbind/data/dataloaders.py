from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from molbind.data.components.tokenizers import SMILES_TOKENIZER, SELFIES_TOKENIZER
import networkx as nx
from networkx import Graph
from typing import List, Dict


MODALITY_DATA_TYPES = {
    "smiles" : str,
    "selfies" : str,
    "graph" : Graph,
    "nmr" : str,
    "ir" : str
}


class StringDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, modality="smiles"):
        super(StringDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.modality = modality
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        for batch in super(StringDataLoader, self).__iter__():
            
            if self.modality == "smiles":
                tokenized_batch = SMILES_TOKENIZER(batch, padding="max_length", truncation=True, return_tensors="pt")
            elif self.modality == "selfies":
                tokenized_batch = SELFIES_TOKENIZER(batch, padding="max_length", truncation=True, return_tensors="pt")
            yield tokenized_batch["input_ids"], tokenized_batch["attention_mask"]


def load_combined_loader(data_modalities : Dict, batch_size : int, shuffle : bool, num_workers : int) -> CombinedLoader:
    loaders = {}
    
    for modality in data_modalities.keys():
        # import pdb; pdb.set_trace()
        if MODALITY_DATA_TYPES[modality] == str:
            loaders[modality] = StringDataLoader(data_modalities[modality], batch_size, shuffle, num_workers, modality)
        elif MODALITY_DATA_TYPES[modality] == Graph:
            loaders[modality] = DataLoader(data_modalities[modality], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return CombinedLoader(loaders, "min_size")


smiles = ["CCO", "CCN", "CCO", "CCN"]
selfies = ["[C][C][O]", "[C][C][N]", "[C][C][O]", "[C][C][N]"]
dummy_graphs = ["dummy_graph", "dummy_graph", "dummy_graph", "dummy_graph"]

combined_loader = load_combined_loader(
    data_modalities = {
    "smiles" : smiles,
    "selfies" : selfies,
    "graph" : dummy_graphs
    }, 
    batch_size=2, 
    shuffle=True,
    num_workers=1)

for batch, batch_idx, dataloader_idx in combined_loader:
    print(f"{batch}, {batch_idx=}, {dataloader_idx=}")