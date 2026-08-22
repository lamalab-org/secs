"""The graph modality: SMILES -> PyG graph -> GIN encoder -> contrastive embedding.

Graphs are the one modality that does not stack into a rectangular tensor, so
what is worth guarding is the seam: the geometric collater, the central-modality
data riding along on the graph, and the reshape back on the model side.
"""

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

from secs.models import GraphGINEncoder, SECSModule
from secs.models.encoders.graph.molclr import (
    GINet,
    num_atom_type,
    num_bond_direction,
    num_bond_type,
    num_chirality_tag,
)
from secs.utils.graph import (
    ATOM_LIST,
    BOND_LIST,
    CHIRALITY_LIST,
    MASK_ATOM_INDEX,
    smiles_to_graph_data,
    smiles_to_masked_graph_views,
)

from secs.data.modalities import ModalityConstants, loader_for
from secs.data.secs_dataset import SECSDataset, columns_to_read

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccccc1", "CCN(CC)CC"]
CONTEXT_LENGTH = 16
# smallest GIN that still exercises message passing, pooling and readout
BACKBONE = {"num_layer": 2, "emb_dim": 32, "feat_dim": 16}


def graph_dataset(smiles: list[str] = SMILES, **columns):
    frame = pd.DataFrame({"smiles": smiles, **columns})
    return SECSDataset(
        data=frame,
        central_modality="smiles",
        other_modalities=["graph"],
        context_length=CONTEXT_LENGTH,
    ).build_datasets_for_modalities()["graph"]


# --- SMILES -> graph -------------------------------------------------------


def test_graph_carries_one_node_per_atom_and_both_bond_directions():
    graph = smiles_to_graph_data("CCO")
    assert graph.x.shape == (3, 2)
    # 2 bonds, each stored in both directions
    assert graph.edge_index.shape == (2, 4)
    assert graph.edge_attr.shape == (4, 2)
    # node features are (atom type index, chirality index); C is atomic number 6
    assert graph.x[:, 0].tolist() == [ATOM_LIST.index(6), ATOM_LIST.index(6), ATOM_LIST.index(8)]


def test_bondless_molecule_still_has_the_shapes_the_encoder_expects():
    """A lone ion has no edges; empty tensors must keep their second dimension."""
    graph = smiles_to_graph_data("[Na+]")
    assert graph.x.shape == (1, 2)
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, 2)


@pytest.mark.parametrize(
    ("label", "smiles"),
    [
        ("dative bond", "[NH3]->[Pt](<-[NH3])(Cl)Cl"),
        ("dummy atom", "*CCO"),
        ("square-planar chirality", "C[Pt@SP1](Cl)(Br)N"),
        ("octahedral chirality", "C[Co@OH1](Cl)(Br)(N)(O)F"),
        ("directional double bond", r"C/C=C/C"),
    ],
)
def test_features_outside_molclr_vocabulary_fold_into_it(label, smiles):
    """Real molecules carry features the pretrained tables have no row for.

    Growing the tables would strand the checkpoint, so these map onto the
    nearest existing row instead. What must never happen is an index past the
    end of an embedding table, which is a crash at forward rather than at
    featurisation.
    """
    graph = smiles_to_graph_data(smiles)
    assert graph is not None, label
    assert graph.x[:, 0].max() < num_atom_type
    assert graph.x[:, 1].max() < num_chirality_tag
    if graph.edge_attr.numel():
        # the last bond row is reserved for the self-loops the encoder adds
        assert graph.edge_attr[:, 0].max() < num_bond_type - 1
        assert graph.edge_attr[:, 1].max() < num_bond_direction


def test_a_dative_bond_is_read_as_a_single_bond():
    """DATIVE is still a two-centre sigma bond, and SINGLE is the nearest row."""
    single = BOND_LIST.index(Chem.rdchem.BondType.SINGLE)
    assert smiles_to_graph_data("O->[Cu]").edge_attr[:, 0].tolist() == [single, single]


def test_chirality_vocabulary_matches_the_encoder_table():
    """Upstream MolCLR lists four tags but sizes the table at three."""
    assert len(CHIRALITY_LIST) == num_chirality_tag


def test_unparseable_smiles_returns_none_rather_than_raising():
    assert smiles_to_graph_data("banana-not-a-smiles") is None


def test_masked_views_keep_the_atoms_and_drop_whole_bonds():
    view_i, view_j = smiles_to_masked_graph_views("CC(=O)Nc1ccccc1")
    full = smiles_to_graph_data("CC(=O)Nc1ccccc1")
    for view in (view_i, view_j):
        assert view.x.shape == full.x.shape
        assert (view.x[:, 0] == MASK_ATOM_INDEX).any()
        # a bond is masked in both of its directions, so the count stays even
        assert view.edge_index.shape[1] % 2 == 0
        assert view.edge_index.shape[1] < full.edge_index.shape[1]


# --- dataset and collation -------------------------------------------------


def test_graph_column_is_derived_from_smiles():
    """No dataset ships a `graph` column; asking for the modality is enough."""
    assert columns_to_read(["graph", "c_nmr"], "smiles") == ["smiles", "c_nmr"]
    # graph as the central modality still only needs the smiles column
    assert columns_to_read(["c_nmr"], "graph") == ["c_nmr", "smiles"]
    assert len(graph_dataset()) == len(SMILES)


def test_graph_can_be_the_central_modality():
    """c_nmr <-> graph directly: two encoders, no SMILES in the model at all."""
    frame = pd.DataFrame({"smiles": SMILES, "c_nmr": [[20.0, 60.0]] * len(SMILES)})
    dataset = SECSDataset(
        data=frame,
        central_modality="graph",
        other_modalities=["c_nmr"],
        context_length=CONTEXT_LENGTH,
    ).build_datasets_for_modalities()["c_nmr"]

    sample = dataset[0]
    assert set(sample) == {"graph", "c_nmr"}
    assert isinstance(sample["graph"], Data)
    shifts, mask = sample["c_nmr"]
    assert shifts.ndim == mask.ndim == 1

    # the pair needs the geometric collater even though c_nmr is the named modality
    batch = next(iter(loader_for("graph", "c_nmr")(dataset, batch_size=3)))
    assert batch["graph"].batch.max() == 2
    assert batch["c_nmr"][0].shape[0] == 3


def test_a_pair_with_a_graph_on_either_side_takes_the_geometric_loader():
    assert loader_for("graph", "c_nmr") is GeometricDataLoader
    assert loader_for("c_nmr", "graph") is GeometricDataLoader
    assert loader_for("c_nmr", "smiles") is TorchDataLoader


def test_samples_look_like_every_other_modality_with_a_graph_inside():
    """Same {central: tokens, modality: sample} shape as the tensor modalities."""
    sample = graph_dataset()[0]
    assert set(sample) == {"smiles", "graph"}
    assert isinstance(sample["graph"], Data)
    input_ids, attention_mask = sample["smiles"]
    assert input_ids.shape == attention_mask.shape == (CONTEXT_LENGTH,)


def test_unparseable_rows_are_dropped_with_their_central_row():
    """The central tokens must follow the surviving rows, not the original ones."""
    smiles = ["CCO", "banana-not-a-smiles", "c1ccccc1O"]
    dataset = graph_dataset(smiles)
    assert len(dataset) == 2
    expected = SECSDataset._tokenize_strings(["CCO", "c1ccccc1O"], "smiles", CONTEXT_LENGTH)[0]
    stacked = torch.stack([dataset[i]["smiles"][0] for i in range(len(dataset))])
    assert torch.equal(stacked, expected)


def test_geometric_loader_batches_graphs_and_central_tokens_together():
    """The collater recurses into the dict: graphs one way, tokens the other."""
    batch = next(iter(GeometricDataLoader(graph_dataset(), batch_size=3)))
    graphs = [smiles_to_graph_data(s) for s in SMILES[:3]]
    assert batch["graph"].x.shape[0] == sum(g.x.shape[0] for g in graphs)
    assert batch["graph"].batch.tolist() == [i for i, g in enumerate(graphs) for _ in range(g.x.shape[0])]
    input_ids, attention_mask = batch["smiles"]
    assert input_ids.shape == attention_mask.shape == (3, CONTEXT_LENGTH)


def test_registry_knows_the_graph_modality():
    assert ModalityConstants["graph"].dataset is type(graph_dataset())


def test_only_graphs_ask_for_the_geometric_loader():
    """Which collater a modality needs follows from its data type, nothing else."""
    assert ModalityConstants["graph"].loader is GeometricDataLoader
    assert ModalityConstants["c_nmr"].loader is TorchDataLoader
    assert ModalityConstants["smiles"].loader is TorchDataLoader


# --- encoder ---------------------------------------------------------------


def batch_of(smiles: list[str] = SMILES):
    """One collated batch, as the loader hands it to the model."""
    return next(iter(GeometricDataLoader(graph_dataset(smiles), batch_size=len(smiles))))


def graphs_of(smiles: list[str] = SMILES):
    """Just the graph side of a batch, for encoder-level tests."""
    return batch_of(smiles)["graph"]


def test_encoder_returns_one_embedding_per_graph():
    encoder = GraphGINEncoder(**BACKBONE).eval()
    with torch.no_grad():
        embedding = encoder(graphs_of())
    assert embedding.shape == (len(SMILES), encoder.output_dim)
    assert encoder.output_dim == BACKBONE["feat_dim"]


def test_projected_readout_halves_the_embedding():
    encoder = GraphGINEncoder(readout="projected", **BACKBONE).eval()
    assert encoder.output_dim == BACKBONE["feat_dim"] // 2
    with torch.no_grad():
        assert encoder(graphs_of()).shape == (len(SMILES), encoder.output_dim)


def test_feat_readout_builds_no_unused_parameters():
    """DDP refuses to step over parameters no loss touched, so don't create any."""
    encoder = GraphGINEncoder(**BACKBONE)
    encoder(graphs_of()).sum().backward()
    assert all(p.grad is not None for p in encoder.parameters())


def test_a_graph_batches_the_same_whatever_it_is_batched_with():
    """Pooling must be per graph: an embedding cannot depend on its neighbours."""
    encoder = GraphGINEncoder(**BACKBONE).eval()
    with torch.no_grad():
        alone = encoder(graphs_of(["CC(=O)Nc1ccccc1"]))
        together = encoder(graphs_of(SMILES))
    assert torch.allclose(alone[0], together[SMILES.index("CC(=O)Nc1ccccc1")], atol=1e-5)


def test_bondless_molecule_survives_the_encoder():
    encoder = GraphGINEncoder(**BACKBONE).eval()
    with torch.no_grad():
        embedding = encoder(graphs_of(["[Na+]", "CCO"]))
    assert embedding.shape == (2, encoder.output_dim)
    assert torch.isfinite(embedding).all()


def test_freezing_stops_gradients_and_pins_eval_mode():
    encoder = GraphGINEncoder(freeze_encoder=True, **BACKBONE)
    assert all(not p.requires_grad for p in encoder.parameters())
    encoder.train()
    assert not encoder._backbone.training
    assert all(not module.training for module in encoder._backbone.modules())


@pytest.mark.parametrize("prefix", ["", "encoder.", "model.encoder."])
def test_checkpoint_round_trip(tmp_path, prefix):
    """A GIN reloads whether it was saved standalone or inside a Lightning run."""
    trained = GraphGINEncoder(**BACKBONE).eval()
    ckpt = tmp_path / "encoder.ckpt"
    torch.save({"state_dict": {f"{prefix}{k}": v for k, v in trained.encoder.state_dict().items()}}, ckpt)

    restored = GraphGINEncoder(ckpt_path=str(ckpt), **BACKBONE).eval()
    with torch.no_grad():
        assert torch.allclose(trained(graphs_of()), restored(graphs_of()), atol=1e-6)


def test_molclr_pth_loads_even_though_it_carries_a_head_we_do_not_build(tmp_path):
    """MolCLR ships a raw .pth of a GINet whose contrastive out_lin we skip."""
    pretrained = GINet(readout="projected", **BACKBONE)
    ckpt = tmp_path / "molclr_gin.pth"
    torch.save(pretrained.state_dict(), ckpt)

    encoder = GraphGINEncoder(ckpt_path=str(ckpt), **BACKBONE)
    assert torch.equal(encoder.encoder.x_embedding1.weight, pretrained.x_embedding1.weight)
    assert torch.equal(encoder.encoder.feat_lin.weight, pretrained.feat_lin.weight)


def test_a_checkpoint_matching_nothing_is_refused(tmp_path):
    """Silently random-initialising behind ckpt_path wastes a whole run."""
    ckpt = tmp_path / "wrong.pth"
    torch.save({"gnn.x_embedding1.weight": torch.randn(119, BACKBONE["emb_dim"])}, ckpt)
    with pytest.raises(ValueError, match=r"nothing in .* matched the backbone"):
        GraphGINEncoder(ckpt_path=str(ckpt), **BACKBONE)


@pytest.mark.parametrize(("kwarg", "value"), [("pool", "median"), ("readout", "logits")])
def test_unknown_backbone_options_are_reported(kwarg, value):
    with pytest.raises(ValueError, match=f"Unknown {kwarg} '{value}'"):
        GINet(**{**BACKBONE, kwarg: value})


# --- the whole path, as the LightningModule sees it ------------------------


@pytest.fixture
def graph_config(stub_encoder_cls):
    return OmegaConf.create(
        {
            "data": {"central_modality": "smiles", "modalities": ["graph"], "batch_size": len(SMILES)},
            "trainer": {"gpus_per_node": 1, "num_nodes": 1},
            "model": {
                "encoders": {"smiles": {"name": "stub"}, "graph": {"name": "molclr", **BACKBONE}},
                "projection_heads": {
                    "smiles_is_on": True,
                    "graph_is_on": True,
                    "smiles_freeze": False,
                    "graph_freeze": False,
                    "smiles": {"dims": [stub_encoder_cls.output_dim, 4], "activation": "LeakyReLU"},
                    "graph": {"dims": [BACKBONE["feat_dim"], 4], "activation": "LeakyReLU"},
                },
                "loss": {"temperature": 0.07, "symmetric": True},
                "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
            },
        }
    )


@pytest.mark.usefixtures("registered_stub")
def test_module_forward_aligns_graphs_with_the_central_modality(graph_config):
    module = SECSModule(graph_config).eval()
    # CombinedLoader hands the step [batch, batch_index, dataloader_index]
    with torch.no_grad():
        embeddings = module((batch_of(), 0, 0))
    assert set(embeddings) == {"smiles", "graph"}
    assert embeddings["smiles"].shape == embeddings["graph"].shape == (len(SMILES), 4)


@pytest.mark.usefixtures("registered_stub")
def test_gradients_reach_every_graph_parameter_through_the_loss(graph_config):
    module = SECSModule(graph_config)
    embeddings = module((batch_of(), 0, 0))
    module._info_nce_loss(embeddings["graph"], embeddings["smiles"]).backward()
    assert all(p.grad is not None for p in module.model.dict_encoders["graph"].parameters())
