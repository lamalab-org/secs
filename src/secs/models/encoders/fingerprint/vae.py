import torch.nn as nn
from torch import Tensor

from secs.models.heads import ProjectionHead


class FingerprintVAEEncoder(nn.Module):
    """MLP variational autoencoder over molecular fingerprints.

    Trained standalone by `FingerprintEncoderModule`; it is not a `ModalityEncoder`
    because it returns (mu, log_var, reconstruction) rather than one embedding.
    """

    def __init__(
        self,
        input_dims: list[int],
        output_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = ProjectionHead(dims=input_dims, activation="leakyrelu")
        # Output layers for mu and log_var
        self.fc_mu = nn.Linear(input_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(input_dims[-1], latent_dim)
        # decoder
        self.decoder = ProjectionHead(dims=output_dims, activation="leakyrelu")

    def encode(self, x: Tensor):
        return self.encoder(x)

    def decode(self, x: Tensor):
        return self.decoder(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        latent_state = self.encode(x)
        mu = self.fc_mu(latent_state)
        log_var = self.fc_log_var(latent_state)
        output = self.decode(latent_state)
        return mu, log_var, output
