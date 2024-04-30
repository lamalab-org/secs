import csv  # noqa: I002
import math
import random
from copy import deepcopy
from typing import List, Tuple  # noqa: UP035

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molbind.data.utils import split_torch_dataset

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def read_smiles(data_path: str) -> List[str]:  # noqa: UP006
    smiles_data = []
    with open(data_path) as csv_file:  # noqa: PTH123
        csv_reader = csv.reader(csv_file, delimiter=",")
        for _, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data[1:]


def smiles_to_graph(smiles: str) -> Tuple[Data, Data]:  # noqa: UP006
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)

    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()

    type_idx = []
    chirality_idx = []
    atomic_number = []

    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append(
            [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        )
        edge_feat.append(
            [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir()),
            ]
        )

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

    # random mask a subgraph of the molecule
    num_mask_nodes = max([1, math.floor(0.25 * N)])
    num_mask_edges = max([0, math.floor(0.25 * M)])
    mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
    mask_nodes_j = random.sample(list(range(N)), num_mask_nodes)
    mask_edges_i_single = random.sample(list(range(M)), num_mask_edges)
    mask_edges_j_single = random.sample(list(range(M)), num_mask_edges)
    mask_edges_i = [2 * i for i in mask_edges_i_single] + [
        2 * i + 1 for i in mask_edges_i_single
    ]
    mask_edges_j = [2 * i for i in mask_edges_j_single] + [
        2 * i + 1 for i in mask_edges_j_single
    ]

    x_i = deepcopy(x)
    for atom_idx in mask_nodes_i:
        x_i[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
    edge_index_i = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
    edge_attr_i = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)
    count = 0
    for bond_idx in range(2 * M):
        if bond_idx not in mask_edges_i:
            edge_index_i[:, count] = edge_index[:, bond_idx]
            edge_attr_i[count, :] = edge_attr[bond_idx, :]
            count += 1
    data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

    x_j = deepcopy(x)
    for atom_idx in mask_nodes_j:
        x_j[atom_idx, :] = torch.tensor([len(ATOM_LIST), 0])
    edge_index_j = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.long)
    edge_attr_j = torch.zeros((2 * (M - num_mask_edges), 2), dtype=torch.long)
    count = 0
    for bond_idx in range(2 * M):
        if bond_idx not in mask_edges_j:
            edge_index_j[:, count] = edge_index[:, bond_idx]
            edge_attr_j[count, :] = edge_attr[bond_idx, :]
            count += 1
    data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
    return data_i, data_j


class MoleculeDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.smiles_data = read_smiles(data_path)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, index: int) -> Tuple:  # noqa: UP006
        return smiles_to_graph(self.smiles_data[index])


def get_train_valid_loaders_from_dataset(
    data_path: str,
    batch_size: int,
    num_workers: int = 0,
    valid_size: float = 0.2,
) -> Tuple[GeometricDataLoader, GeometricDataLoader]:  # noqa: UP006
    """Generate a torch DataLoader from a dataset.

    Args:
        data_path (str): path to data
        batch_size (int): batch size
        shuffle (bool, optional): shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.
    Returns:
        Tuple[GeometricDataLoader, GeometricDataLoader]: train and validation dataloaders as tuple
    """
    dataset = MoleculeDataset(data_path)
    # split dataset into train and test
    train_dataset, val_dataset = split_torch_dataset(dataset, valid_size)

    train_loader = GeometricDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = GeometricDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader
