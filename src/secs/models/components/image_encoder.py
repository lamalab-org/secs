from typing import List

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor


class MolScribeImageEncoder(nn.Module):
    """Molecule-image encoder built on MolScribe's Swin-B backbone.

    Wraps the MolScribe image encoder and mean-pools its patch features into a
    single fixed-size embedding (dim ``n_features`` == 1024 for ``swin_base``),
    matching the ``forward(x) -> [B, H]`` contract of the other MolBind
    encoders (e.g. :class:`SmilesEncoder`).

    Args:
        ckpt_path: local path to a MolScribe checkpoint. If ``None`` (default)
            the checkpoint is downloaded from the HuggingFace Hub.
        repo_id / checkpoint_filename: HuggingFace Hub location used when
            ``ckpt_path`` is ``None``.
        freeze_encoder: if ``True``, freeze the backbone weights.
        pretrained: if ``True`` load the MolScribe weights; if ``False`` keep
            the architecture but re-initialize the weights (train from scratch).
    """

    def __init__(
        self,
        ckpt_path: str | None = None,
        repo_id: str = "yujieq/MolScribe",
        checkpoint_filename: str = "swin_base_char_aux_1m.pth",
        freeze_encoder: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.pretrained = pretrained
        self._initialize_encoder(ckpt_path, repo_id, checkpoint_filename)

    def _initialize_encoder(self, ckpt_path: str | None, repo_id: str, checkpoint_filename: str) -> None:
        # Imported lazily so the project does not hard-depend on molscribe.
        from molscribe import MolScribe

        if ckpt_path is None:
            from huggingface_hub import hf_hub_download

            ckpt_path = hf_hub_download(repo_id, checkpoint_filename)

        # MolScribe builds both encoder and decoder + loads weights; we keep
        # only the encoder (and the image transform) and let the rest be freed.
        molscribe = MolScribe(ckpt_path, device=torch.device("cpu"))
        self.encoder = molscribe.encoder
        # albumentations transform that turns an HxWx3 uint8 image into the
        # 384x384 normalized tensor the backbone expects (useful for datasets).
        self.transform = molscribe.transform
        self.n_features = self.encoder.n_features

        if not self.pretrained:
            for param in self.encoder.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            logger.info("MolScribeImageEncoder: re-initialized backbone weights (pretrained=False)")

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: tuple[Tensor, ...] | Tensor) -> Tensor:
        # Accept either a bare image tensor [B, 3, 384, 384] or a tuple whose
        # first element is that tensor (to match the project's encoder calling
        # convention).
        image_tensor = x[0] if isinstance(x, (tuple, list)) else x
        features, _ = self.encoder(image_tensor)  # features: [B, S, H]
        return features.mean(dim=1)  # mean-pool over patches -> [B, H]


class ImageEncoder(nn.Module):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        self.cfg = [
            [128, 7, 3, 4],
            [256, 5, 1, 1],
            [384, 5, 1, 1],
            "M",
            [384, 3, 1, 1],
            [384, 3, 1, 1],
            "M",
            [512, 3, 1, 1],
            [512, 3, 1, 1],
            [512, 3, 1, 1],
            "M",
        ]
        self.features = self.make_layers()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        if ckpt_path is not None:
            self.model.load_state_dict(torch.load(ckpt_path), strict=False)

    def make_layers(self, batch_norm: bool = False) -> nn.Sequential:
        """
        :param batch_norm: boolean of batch normalization should be used in-between conv2d and relu activation.
                        Defaults to False
        :return: torch.nn.Sequential module as feature-extractor
        """
        layers: List[nn.Module] = []  # noqa: UP006

        in_channels = 1
        for v in self.cfg:
            if v == "A":
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                if v == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    units, kern_size, stride, padding = v
                    conv2d = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=units,
                        kernel_size=kern_size,
                        stride=stride,
                        padding=padding,
                    )
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(units), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = units
        return nn.Sequential(*layers)

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        x = self.features(x)
        return self.flatten(self.pool(x))
