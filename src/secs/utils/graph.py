"""SMILES -> PyG graphs, with the atom/bond vocabulary MolCLR was pretrained on.

`smiles_to_graph_data` builds the plain molecular graph the `graph` modality is
trained on; `smiles_to_masked_graph_views` builds the two randomly masked views
MolCLR-style contrastive *pretraining* of the GNN uses.
"""

import math
import random
from copy import deepcopy

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]

#: index the encoder's atom embedding reserves for a masked atom
MASK_ATOM_INDEX = len(ATOM_LIST)


def mol_to_graph_tensors(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, edge_index, edge_attr) for one RDKit molecule.

    Nodes carry (atom type index, chirality index); every bond becomes the two
    directed edges a message-passing net needs, each with (bond type, bond
    direction).
    """
    type_idx, chirality_idx = [], []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x = torch.tensor([type_idx, chirality_idx], dtype=torch.long).t().contiguous()

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
        row += [start, end]
        col += [end, start]
        edge_feat += [feat, feat]

    # A bond-less molecule (a lone atom, a salt fragment) still has to come out
    # with the shapes the collater and the GNN expect.
    edge_index = torch.tensor([row, col], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(np.array(edge_feat, dtype=np.int64), dtype=torch.long).reshape(-1, 2)
    return x, edge_index, edge_attr


def smiles_to_graph_data(smiles: str) -> Data | None:
    """The molecular graph for one SMILES, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x, edge_index, edge_attr = mol_to_graph_tensors(mol)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _masked_view(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_mask_nodes: int,
    num_mask_edges: int,
) -> Data:
    """One MolCLR view: mask a random subset of atoms, drop a random subset of bonds."""
    num_nodes, num_bonds = x.size(0), edge_index.size(1) // 2

    x = deepcopy(x)
    for atom_idx in random.sample(range(num_nodes), num_mask_nodes):
        x[atom_idx, :] = torch.tensor([MASK_ATOM_INDEX, 0])

    masked_bonds = set(random.sample(range(num_bonds), num_mask_edges))
    # both directed edges of a masked bond go away together
    keep = [i for i in range(2 * num_bonds) if i // 2 not in masked_bonds]
    return Data(x=x, edge_index=edge_index[:, keep], edge_attr=edge_attr[keep, :])


def smiles_to_masked_graph_views(smiles: str, mask_ratio: float = 0.25) -> tuple[Data, Data]:
    """Two independently masked views of the same molecule (MolCLR pretraining)."""
    mol = Chem.MolFromSmiles(smiles)
    x, edge_index, edge_attr = mol_to_graph_tensors(mol)

    num_mask_nodes = max(1, math.floor(mask_ratio * mol.GetNumAtoms()))
    num_mask_edges = max(0, math.floor(mask_ratio * mol.GetNumBonds()))
    return (
        _masked_view(x, edge_index, edge_attr, num_mask_nodes, num_mask_edges),
        _masked_view(x, edge_index, edge_attr, num_mask_nodes, num_mask_edges),
    )
