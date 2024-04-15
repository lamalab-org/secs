"""
This module contains the implementation of the FingerprintEncoderModule class, which is a PyTorch Lightning module.
The FingerprintEncoderModule class is used to train the FingerprintEncoder model using PyTorch Lightning.
"""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor

from molbind.models.components.base_encoder import FingerprintEncoder


class FingerprintEncoderModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = FingerprintEncoder(cfg=cfg)
        self.config = cfg

    def forward(self, input_fingerprint: Tensor):
        return self.model(input_fingerprint)

    def _vae_loss(self, input_fingerprint, prefix="train"):
        # Reconstruction loss
        if self.current_epoch > 5:
            mu, log_var, output_fingerprint = self.model(input_fingerprint)
            recon_loss = F.mse_loss(output_fingerprint, input_fingerprint)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))
            self.log(f"recon_loss_{prefix}", recon_loss)
            self.log(f"kl_loss_{prefix}", kl_loss)
        else:
            kl_loss = torch.tensor(0)
            mu, log_var, output_fingerprint = self.model(input_fingerprint)
            recon_loss = F.mse_loss(output_fingerprint, input_fingerprint)
        return torch.mean(recon_loss + kl_loss)

    def training_step(self, fingerprints: Tensor):
        return self._vae_loss(input_fingerprint=fingerprints, prefix="train")

    def validation_step(self, fingerprints: Tensor):
        return self._vae_loss(input_fingerprint=fingerprints, prefix="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )
