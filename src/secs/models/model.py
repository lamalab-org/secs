import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor

from secs.models import ProjectionHead


class MolBind(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        # Deferred: secs.data.modalities imports the encoder classes from
        # secs.models, so importing it at module level would be circular.
        from secs.data.modalities import ModalityConstants  # noqa: PLC0415

        modalities = cfg.data.modalities
        central_modality = cfg.data.central_modality
        self.central_modality = central_modality
        logger.info(f"Non-central modalities: {modalities}")

        # Instantiate all encoders and projection heads
        dict_encoders, dict_projection_heads = {}, {}
        # Add other modalities to `dict_encoders` and `dict_projection_heads`
        for modality in [*modalities, central_modality]:
            if modality not in [*vars(ModalityConstants)]:
                raise ValueError(f"Modality {modality} not supported yet.")
            dict_encoders[modality] = ModalityConstants[modality].encoder(**cfg.model.encoders[modality])

            if cfg.model.projection_heads[f"{modality}_is_on"]:
                dict_projection_heads[modality] = ProjectionHead(**cfg.model.projection_heads[modality])

        # convert dicts to nn.Moduledict
        self.dict_encoders = nn.ModuleDict(dict_encoders)
        self.dict_projection_heads = nn.ModuleDict(dict_projection_heads)

        # add requires grad to projection heads
        for modality, projection_head in self.dict_projection_heads.items():
            if cfg.model.projection_heads[f"{modality}_freeze"]:
                for param in projection_head.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_data: dict[str, tuple[Tensor, Tensor] | Tensor],
    ) -> dict[str, Tensor]:
        # Lightning hands over [data, batch_index, dataloader_index]
        if isinstance(input_data, tuple):
            input_data, _, _ = input_data

        modality = next(key for key in input_data if key != self.central_modality)
        return {name: self.encode_modality(input_data[name], name) for name in (self.central_modality, modality)}

    def encode_modality(self, input_data: Tensor | tuple[Tensor, Tensor], modality: str) -> Tensor:
        """Encode one modality, applying its projection head if it has one."""
        embedding = self.dict_encoders[modality](input_data)
        if modality in self.dict_projection_heads:
            embedding = self.dict_projection_heads[modality](embedding)
        return embedding
