import os  # noqa: I002

import pytorch_lightning as L
from dotenv import load_dotenv
from lightning import Trainer
from omegaconf import DictConfig

from molbind.data.utils.graph_utils import MoleculeDatasetWrapper
from molbind.models.components.molclr_lightningmodule import GCNModule

load_dotenv(".env")

def train_molclr():
    # loss device, batch_size, temperature, use_cosine_similarity
    """
    References:
        config ref: https://github.com/yuyangw/MolCLR/blob/master/config.yaml
        data   ref: https://codeocean.com/capsule/6901415/tree/v1
        model  ref: https://github.com/yuyangw/MolCLR/blob/master/models/gcn_molclr.py
    """
    cfg = {
        "model": {
            "num_layer": 5,
            "drop_ratio": 0,
            "feat_dim": 512,
            "pool": "mean",
            "emb_dim": 300,
        },
        "data": {
            "batch_size": 512,
            "num_workers": 4,
            "data_path": "../data/pretrain_example.csv",
            "valid_size": 0.2,
        },
        "trainer": {
            "max_epochs": 100,
            "log_every_n_steps": 5,
            "accelerator": "mps",
            "devices": 1,
        },
        "loss": {
            "device": "mps",
            "batch_size": 512,
            "temperature": 0.1,
            "use_cosine_similarity": True,
        },
        "optimizer": {
            "lr": 0.0005,
            "weight_decay": 1e-5,
        },
    }
    # convert to DictConfig
    cfg = DictConfig(cfg)

    train_dataloader, val_dataloader = MoleculeDatasetWrapper(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_size=cfg.data.valid_size,
        data_path=cfg.data.data_path,
    ).get_data_loaders()

    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
    )

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=wandb_logger,
        devices=cfg.trainer.devices,
    )
    # set-up model
    trainer.fit(
        model=GCNModule(cfg),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

if __name__ == "__main__":
    train_molclr()
