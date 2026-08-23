"""Per-row access to the central modality, whatever kind of thing it is.

Every paired dataset stores the central modality alongside its own and hands
one row of it back per sample. What that row *is* differs: tokenized SMILES are
a pair of tensors, a molecular graph is a `Data`. These views keep that
difference in one place, so a dataset only ever asks for "row i of the central
modality" and never has to know which.
"""

from torch import Tensor
from torch_geometric.data import Data

from secs.utils.graph import smiles_to_graph_data


class CentralModalityData:
    """Row-addressable central modality data, restrictable to a subset of rows."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def select(self, rows: list[int]) -> "CentralModalityData":
        raise NotImplementedError


class TokenizedCentralModality(CentralModalityData):
    """A tokenized string modality: (input_ids, attention_mask) per row."""

    def __init__(self, input_ids: Tensor, attention_mask: Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> list[Tensor]:
        return [self.input_ids[index], self.attention_mask[index]]

    def select(self, rows: list[int]) -> "TokenizedCentralModality":
        return TokenizedCentralModality(self.input_ids[rows], self.attention_mask[rows])


class GraphCentralModality(CentralModalityData):
    """A molecular graph per row, built from SMILES in the worker process.

    Kept as strings rather than prebuilt graphs for the same reason as
    `GraphDataset`: at a million molecules the tensors would not be worth their
    memory, and the conversion parallelises across dataloader workers.
    """

    def __init__(self, smiles: list[str]) -> None:
        self.smiles = smiles

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Data:
        return smiles_to_graph_data(self.smiles[index])

    def select(self, rows: list[int]) -> "GraphCentralModality":
        return GraphCentralModality([self.smiles[row] for row in rows])
