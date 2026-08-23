"""Checkpoints written by SECSModule must reload into SECSModule without key surgery."""

import pytest
import torch

from secs.models import SECSModule


def batch():
    return {
        "smiles": (torch.randint(0, 64, (4, 5)), torch.ones(4, 5, dtype=torch.long)),
        "c_nmr": (torch.rand(4, 7) * 200, torch.ones(4, 7, dtype=torch.bool)),
    }


def save(module: SECSModule, path) -> str:
    torch.save({"state_dict": module.state_dict()}, path)
    return str(path)


def test_checkpoint_reloads_without_renaming_keys(registered_stub, secs_config, tmp_path):
    """The keys are already relative to the module, so nothing has to be stripped."""
    trained = SECSModule(secs_config(registered_stub))
    ckpt = save(trained, tmp_path / "secs.ckpt")

    restored = SECSModule(secs_config(registered_stub, ckpt_path=ckpt))

    for (name, saved), (_, loaded) in zip(trained.state_dict().items(), restored.state_dict().items(), strict=True):
        assert torch.equal(saved, loaded), f"{name} was not restored"


def test_restored_module_reproduces_the_embeddings(registered_stub, secs_config, tmp_path):
    trained = SECSModule(secs_config(registered_stub)).eval()
    restored = SECSModule(secs_config(registered_stub, ckpt_path=save(trained, tmp_path / "secs.ckpt"))).eval()

    inputs = batch()
    with torch.no_grad():
        before, after = trained(inputs), restored(inputs)
    for modality in before:
        assert torch.allclose(before[modality], after[modality], atol=1e-6)


def test_learnable_temperature_survives_the_round_trip(registered_stub, secs_config, tmp_path):
    """The temperature is a parameter of the LightningModule, not of MolBind.

    Loading into `self.model` used to drop it; loading into `self` keeps it.
    """
    loss_cfg = {"model": {"loss": {"learnable_temperature": True}}}
    trained = SECSModule(secs_config(registered_stub, **loss_cfg))
    with torch.no_grad():
        trained.log_inv_temperature.fill_(2.5)

    restored = SECSModule(secs_config(registered_stub, ckpt_path=save(trained, tmp_path / "secs.ckpt"), **loss_cfg))
    assert restored.log_inv_temperature.item() == 2.5


def test_hydra_none_checkpoint_means_from_scratch(registered_stub, secs_config):
    """Hydra configs spell "no checkpoint" as a bare None, which YAML reads as a string."""
    module = SECSModule(secs_config(registered_stub, ckpt_path="None"))
    assert isinstance(module, SECSModule)


def test_missing_checkpoint_does_not_stop_training(registered_stub, secs_config, tmp_path):
    module = SECSModule(secs_config(registered_stub, ckpt_path=str(tmp_path / "nope.ckpt")))
    assert isinstance(module, SECSModule)
