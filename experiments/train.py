from molbind.models.lightning_module import train_molbind
from omegaconf import DictConfig


if __name__ == "__main__":
    config = {
        "wandb": {"entity": "adrianmirza", "project_name": "embedbind"},
        "model": {
            "projection_heads": {
                "selfies": {"dims": [256, 128], "activation": "leakyrelu", "batch_norm": False},
                "smiles": {"dims": [256, 128], "activation": "leakyrelu", "batch_norm": False},
            },
            "encoders": {
                "smiles": {"pretrained": True, "freeze_encoder": False},
                "selfies": {"pretrained": True, "freeze_encoder": False},
            },
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
        },
        "loss": {"temperature": 0.1},
        "data": {
            "central_modality": "smiles",
            "modalities": ["selfies"],
            "dataset_path": "subset.csv",
            "train_frac": 0.8,
            "valid_frac": 0.2,
            "seed": 42,
            "fraction_data": 1,
            "batch_size": 64,
        },
    }

    config = DictConfig(config)
    train_molbind(config)
