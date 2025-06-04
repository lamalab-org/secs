"""
This module contains the implementation of the FingerprintEncoderModule class, which is a PyTorch Lightning module.
The FingerprintEncoderModule class is used to train the FingerprintEncoder model using PyTorch Lightning.
Modified to handle two different fingerprint types as input and target.
"""

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import Tensor

from molbind.models.components.base_encoder import FingerprintEncoder


class FingerprintEncoderModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.model = FingerprintEncoder(
            input_dims=cfg.model.input_dims,
            output_dims=cfg.model.output_dims,
            latent_dim=cfg.model.latent_dim,
        )
        self.config = cfg
        self.beta = cfg.model.loss.beta_kl_loss
        self.batch_size = cfg.data.batch_size

    def forward(self, input_fingerprint: Tensor) -> Tensor:
        return self.model(input_fingerprint)

    def _vae_loss(self, input_fingerprint: Tensor, target_fingerprint: Tensor, prefix="train") -> Tensor:
        """
        VAE loss function for cross-fingerprint reconstruction.

        Args:
            input_fingerprint: Source fingerprint (encoder input)
            target_fingerprint: Target fingerprint (reconstruction target)
            prefix: Logging prefix for metrics

        Returns:
            Total loss value

        References:
        https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
        KL-annealing: https://arxiv.org/pdf/1511.06349.pdf
        """
        mu, log_var, output_fingerprint = self.model(input_fingerprint)

        # Reconstruction loss (comparing output to target fingerprint)
        recon_loss = F.mse_loss(output_fingerprint, target_fingerprint)

        # KL divergence loss (regularization term)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))

        # Apply KL annealing during warmup
        kl_weight = 0.0 if self.current_epoch <= self.config.model.warmup_epochs else self.beta

        total_loss = recon_loss + kl_weight * kl_loss

        # Logging
        self.log(f"recon_loss_{prefix}", recon_loss)
        self.log(f"kl_loss_{prefix}", kl_loss)
        self.log(f"total_loss_{prefix}", total_loss)

        # Calculate reconstruction accuracy (fraction of correctly reconstructed bits)
        with torch.no_grad():
            output_fingerprint_rounded = torch.round(output_fingerprint)
            correct_recon = torch.sum(output_fingerprint_rounded == target_fingerprint).item()
            total_elements = target_fingerprint.numel()
            accuracy = correct_recon / total_elements
            self.log(f"correct_recon_{prefix}", accuracy)

        return total_loss

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """
        Training step expecting a tuple of (input_fingerprint, target_fingerprint)
        """
        input_fingerprint, target_fingerprint = batch
        return self._vae_loss(input_fingerprint=input_fingerprint, target_fingerprint=target_fingerprint, prefix="train")

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """
        Validation step expecting a tuple of (input_fingerprint, target_fingerprint)
        """
        input_fingerprint, target_fingerprint = batch
        return self._vae_loss(input_fingerprint=input_fingerprint, target_fingerprint=target_fingerprint, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )


if __name__ == "__main__":
    dataset_path = "/home/mirzaa/projects/MoleculeBind/data/adapt_sim_to_exp/joint_dataset.parquet"
    # cols hnmr_exp and hnmr_sim
    import pandas as pd
    from torch.utils.data import DataLoader, Dataset

    class JointDataset(Dataset):
        def __init__(self, dataset_path: str):
            self.data = pd.read_parquet(dataset_path)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            input_fingerprint = torch.tensor(self.data.iloc[idx]["h_nmr_sim"], dtype=torch.float32)
            target_fingerprint = torch.tensor(self.data.iloc[idx]["h_nmr_exp"], dtype=torch.float32)
            return input_fingerprint, target_fingerprint

    dataset = JointDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "model": {
                "input_dims": [10000, 1024],
                "output_dims": [1024, 10000],
                "latent_dim": 64,
                "loss": {"beta_kl_loss": 0.1},
                "warmup_epochs": 5,
                "optimizer": {"lr": 0.0001, "weight_decay": 1e-5},
            },
            "data": {"batch_size": 16},
        }
    )

    model = FingerprintEncoderModule(cfg)
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    trainer = Trainer(
        max_epochs=25,
        accelerator="gpu",
        devices=1,
        logger=WandbLogger(project="fp_vae_lightning", name="fp_vae_experiment")
    )
    trainer.fit(model, dataloader)
