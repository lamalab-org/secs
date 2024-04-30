import os  # noqa: I002

import pytorch_lightning as L
from dotenv import load_dotenv
from lightning import Trainer
from omegaconf import DictConfig

from molbind.data.utils.graph_utils import get_train_valid_loaders_from_dataset
from molbind.models.components.molclr_lightningmodule import GCNModule
from molbind.utils.utils import select_device

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
            "accelerator": select_device(),
            "devices": 1,
        },
        "loss": {
            "temperature": 0.1,
        },
        "optimizer": {
            "lr": 0.0005,
            "weight_decay": 1e-5,
        },
    }
    # convert to DictConfig
    cfg = DictConfig(cfg)

    train_dataloader, val_dataloader = get_train_valid_loaders_from_dataset(
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_size=cfg.data.valid_size,
    )

    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
    )

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=select_device(),
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
