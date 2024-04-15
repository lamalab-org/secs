# Description: Train a VAE on the fingerprint dataset
import os
import pickle as pkl

import numpy as np
import pytorch_lightning as L
import torch
from dotenv import load_dotenv
from lightning import Trainer
from omegaconf import DictConfig

from molbind.data.components.datasets import FingerprintVAEDataset as FingerprintDataset
from molbind.models.components.fp_vae_lightningmodule import FingerprintEncoderModule

if __name__ == "__main__":
    with open("fingerprint.pkl", "rb") as f:
        data = pkl.load(f)

    data = data.sample(frac=1, random_state=42)
    fingerprints = data["fingerprint"].to_list()
    fingerprints = np.vstack(fingerprints).astype(np.float32)
    # load the data

    load_dotenv(".env")

    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="mps",
        log_every_n_steps=5,
        logger=wandb_logger,
        devices=1,
    )

    val_dataset = FingerprintDataset(fingerprints[:1000])
    train_dataset = FingerprintDataset(fingerprints[1000:])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=1, drop_last=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=1, drop_last=True
    )

    cfg = {
        "model": {
            "input_dim": [2048, 1024, 512],
            "output_dim": [512, 512, 2048],
            "latent_dim": 512,
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
            },
        }
    }
    cfg = DictConfig(cfg)
    trainer.fit(
        model=FingerprintEncoderModule(cfg),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
