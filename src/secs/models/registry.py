"""Registry mapping (modality, backbone name) -> encoder class.

Each modality can offer several backbones; the config picks one by name:

    model:
      encoders:
        c_nmr:
          name: transformer      # optional, falls back to the default below
          freeze_encoder: False

Backbones are named after their architecture ("cnn", "transformer") or, for
pretrained checkpoints, after the model itself ("molformer").
"""

from collections.abc import Callable

ENCODER_REGISTRY: dict[str, dict[str, type]] = {}
DEFAULT_BACKBONE: dict[str, str] = {}


def register_encoder(modality: str, name: str, default: bool = False) -> Callable[[type], type]:
    """Class decorator registering an encoder as backbone `name` of `modality`."""

    def decorator(cls: type) -> type:
        backbones = ENCODER_REGISTRY.setdefault(modality, {})
        if name in backbones and backbones[name] is not cls:
            raise ValueError(f"Backbone '{name}' already registered for modality '{modality}'.")
        backbones[name] = cls
        if default or modality not in DEFAULT_BACKBONE:
            DEFAULT_BACKBONE[modality] = name
        return cls

    return decorator


def resolve_encoder(modality: str, name: str | None = None) -> type:
    """Look up the encoder class for a modality, defaulting to its default backbone."""
    if modality not in ENCODER_REGISTRY:
        raise ValueError(f"No encoder registered for modality '{modality}'. Known: {sorted(ENCODER_REGISTRY)}")
    backbones = ENCODER_REGISTRY[modality]
    name = name or DEFAULT_BACKBONE[modality]
    if name not in backbones:
        raise ValueError(f"Unknown backbone '{name}' for modality '{modality}'. Available: {sorted(backbones)}")
    return backbones[name]


def available_encoders() -> dict[str, list[str]]:
    return {modality: sorted(backbones) for modality, backbones in ENCODER_REGISTRY.items()}
