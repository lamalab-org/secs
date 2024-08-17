import csv
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch import tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.utils import one_hot
from torch_scatter import scatter

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


def read_smiles(data_path: str) -> list[str]:
    smiles_data = []
    with Path(data_path).open("r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for _, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data[1:]


def construct_graph(smiles: str) -> tuple[tensor, tensor, tensor, int, int]:
    """_summary_

    Args:
        smiles (str): smiles string

    Returns:
        tuple[tensor, tensor, tensor, int, int]: returns graph data and number of atoms and bonds
    """
    mol = Chem.MolFromSmiles(smiles)
    nr_of_atoms = mol.GetNumAtoms()
    nr_of_bonds = mol.GetNumBonds()
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
    return x, edge_index, edge_attr, nr_of_atoms, nr_of_bonds


def smiles_to_graph_without_augment(smiles: str) -> Data:
    x, edge_index, edge_attr, _, _ = construct_graph(smiles)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def smiles_to_graph(smiles: str) -> tuple[Data, Data]:
    x, edge_index, edge_attr, N, M = construct_graph(smiles)
    # random mask a subgraph of the molecule
    # start augmenting the data
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

    def __getitem__(self, index: int) -> tuple:
        return smiles_to_graph(self.smiles_data[index])


def get_train_valid_loaders_from_dataset(
    data_path: str,
    batch_size: int,
    num_workers: int = 0,
    valid_size: float = 0.2,
) -> tuple[GeometricDataLoader, GeometricDataLoader]:
    """Generate a torch DataLoader from a dataset.

    Args:
        data_path (str): path to data
        batch_size (int): batch size
        shuffle (bool, optional): shuffle data. Defaults to True.
        num_workers (int, optional): number of workers. Defaults to 0.
    Returns:
        tuple[GeometricDataLoader, GeometricDataLoader]: train and validation dataloaders as tuple
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


def get_item_for_dimenet(
    sdf_file: str,
    # y: tensor,
    i: int,  # precomputed_coords: bool = True
) -> Data:
    types = {
        "H": 0,
        "C": 1,
        "N": 2,
        "O": 3,
        "F": 4,
        "P": 5,
        "S": 6,
        "Cl": 7,
        "Br": 8,
        "I": 9,
    }
    bonds = {
        bond_type: i
        for i, bond_type in enumerate([BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC])
    }
    # mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)  # Add hydrogens
    # AllChem.EmbedMolecule(mol)  # Generate 3D conformer
    # AllChem.UFFOptimizeMolecule(mol)  # Optimize conformer
    read_sdf_from_file = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    mol = read_sdf_from_file[0]
    y_in_eh = mol.GetProp("DFT:HOMO_LUMO_GAP")
    # convert hartree to eV
    y = torch.tensor(float(y_in_eh) * 27.2114)
    N = mol.GetNumAtoms()
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = torch.tensor(pos, dtype=torch.float)

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    for atom in mol.GetAtoms():
        type_idx.append(types[atom.GetSymbol()])
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    edge_attr = one_hot(edge_type, num_classes=len(bonds))

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

    x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = (
        torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs], dtype=torch.float)
        .t()
        .contiguous()
    )
    x = torch.cat([x1, x2], dim=-1)
    return Data(
        x=x,
        z=z,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        file=sdf_file,
        idx=i,
    )
