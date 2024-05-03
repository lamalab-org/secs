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
    with open("../data/fingerprint.pkl", "rb") as f:  # noqa: PTH123
        data = pkl.load(f)

    cfg = {
        "model": {
            "input_dims": [2048, 1024, 768],
            "output_dims": [768, 1024, 2048],
            "latent_dim": 512,
            "optimizer": {
                "lr": 1e-4,
                "weight_decay": 1e-5,
            },
            "loss": {
                "beta_kl_loss": 1e-3,
            },
        },
        "trainer": {
            "max_epochs": 100,
            "log_every_n_steps": 5,
            "accelerator": "gpu",
            "devices": 1,
        },
        "data": {
            "batch_size": 128,
        },
        "warmup_epochs": 10 # TODO: no idea what you used, Adrian
    }
    cfg = DictConfig(cfg)

    data = data.sample(frac=1, random_state=42)
    fingerprints = data["fingerprint"].to_list()
    fingerprints = np.vstack(fingerprints).astype(np.float32)
    # load the data

    load_dotenv(".env")

    #wandb_logger = L.loggers.WandbLogger(
    #    project=os.getenv("WANDB_PROJECT"),
    #    entity=os.getenv("WANDB_ENTITY"),
    #)

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        #logger=wandb_logger,
        devices=cfg.trainer.devices,
    )

    val_dataset = FingerprintDataset(fingerprints[:500])
    train_dataset = FingerprintDataset(fingerprints[500:])
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=1, drop_last=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, drop_last=True
    )

    trainer.fit(
        model=FingerprintEncoderModule(cfg),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
