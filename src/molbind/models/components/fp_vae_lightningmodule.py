"""
This module contains the implementation of the FingerprintEncoderModule class, which is a PyTorch Lightning module.
The FingerprintEncoderModule class is used to train the FingerprintEncoder model using PyTorch Lightning.
"""  # noqa: I002

import torch
import torch.nn.functional as F
from lightning import LightningModule
from omegaconf import DictConfig
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

    def _vae_loss(self, input_fingerprint, prefix="train") -> Tensor:
        """
        https://stats.stackexchange.com/questions/341954/balancing-reconstruction-vs-kl-loss-variational-autoencoder
        KL-annealing: https://arxiv.org/pdf/1511.06349.pdf
        """
        if self.current_epoch > self.config.warmup_epochs:
            mu, log_var, output_fingerprint = self.model(input_fingerprint)
            # Reconstruction loss
            recon_loss = F.mse_loss(output_fingerprint, input_fingerprint)
            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))
            total_loss = recon_loss + self.beta * kl_loss
        else:
            kl_loss = torch.tensor(0)
            mu, log_var, output_fingerprint = self.model(input_fingerprint)
            recon_loss = F.mse_loss(output_fingerprint, input_fingerprint)
            total_loss = recon_loss + self.beta * kl_loss
        self.log(f"recon_loss_{prefix}", recon_loss)
        self.log(f"kl_loss_{prefix}", kl_loss)
        self.log(f"total_loss_{prefix}", total_loss)
        # how many bits are correctly reconstructed (fraction)
        # round output fingerprint
        output_fingerprint = torch.round(output_fingerprint)
        correct_recon = torch.sum(output_fingerprint == input_fingerprint).item()
        self.log(
            f"correct_recon_{prefix}",
            correct_recon / self.batch_size / self.config.model.input_dims[0],
        )
        return torch.mean(recon_loss + self.beta * kl_loss)

    def training_step(self, fingerprints: Tensor) -> Tensor:
        return self._vae_loss(input_fingerprint=fingerprints, prefix="train")

    def validation_step(self, fingerprints: Tensor) -> Tensor:
        return self._vae_loss(input_fingerprint=fingerprints, prefix="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )
