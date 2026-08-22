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
from loguru import logger
from rdkit import Chem
from torch_geometric.data import Data

# These four vocabularies are not ours to grow: they are the rows of MolCLR's
# pretrained embedding tables (119 atoms, 3 chiral tags, 5 bond types with the
# last reserved for self-loops, 3 bond directions). Real molecules carry
# features outside them -- dative bonds, dummy atoms, square-planar chirality --
# so anything unlisted maps onto the nearest row that exists instead of growing
# the table, which would strand the checkpoint.
ATOM_LIST = list(range(1, 119))
#: Only the three tags MolCLR's table has rows for. Upstream MolCLR lists a
#: fourth (CHI_OTHER) while sizing the table at three, so a molecule carrying it
#: indexes out of bounds at forward; everything past these three folds into
#: CHI_UNSPECIFIED here.
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
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


MASK_ATOM_INDEX = len(ATOM_LIST)

_ATOM_FALLBACK = (MASK_ATOM_INDEX, "the reserved unknown-atom row")
_CHIRALITY_FALLBACK = (0, "CHI_UNSPECIFIED")
_BOND_FALLBACK = (BOND_LIST.index(Chem.rdchem.BondType.SINGLE), "SINGLE")
_BONDDIR_FALLBACK = (0, "BondDir.NONE")

_WARNED_UNSUPPORTED: set = set()


def _index_or(vocabulary: list, value, fallback: tuple[int, str], what: str) -> int:
    """Row for `value`, or the named `fallback` row if the table has none for it."""
    try:
        return vocabulary.index(value)
    except ValueError:
        index, name = fallback
        if value not in _WARNED_UNSUPPORTED:
            _WARNED_UNSUPPORTED.add(value)
            logger.warning(f"{what} {value} is outside MolCLR's vocabulary; mapping it to {name}.")
        return index


def mol_to_graph_tensors(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, edge_index, edge_attr) for one RDKit molecule.

    Nodes carry (atom type index, chirality index); every bond becomes the two
    directed edges a message-passing net needs, each with (bond type, bond
    direction).
    """
    type_idx, chirality_idx = [], []
    for atom in mol.GetAtoms():
        type_idx.append(_index_or(ATOM_LIST, atom.GetAtomicNum(), _ATOM_FALLBACK, "Atomic number"))
        chirality_idx.append(_index_or(CHIRALITY_LIST, atom.GetChiralTag(), _CHIRALITY_FALLBACK, "Chiral tag"))

    x = torch.tensor([type_idx, chirality_idx], dtype=torch.long).t().contiguous()

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [
            _index_or(BOND_LIST, bond.GetBondType(), _BOND_FALLBACK, "Bond type"),
            _index_or(BONDDIR_LIST, bond.GetBondDir(), _BONDDIR_FALLBACK, "Bond direction"),
        ]
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
