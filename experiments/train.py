from molbind.models.lightning_module import train_molbind


if __name__ == "__main__":
    config = {
        "wandb": {"entity": "wandb_username", "project_name": "embedbind"},
        "model": {
            "projection_heads": {
                "selfies": {"dims": [256, 128]},
                "smiles": {"dims": [256, 128]},
            }
        },
        "loss": {"temperature": 0.1},
        "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
        "data": {
            "modalities": ["selfies"],
            "dataset_path": "subset.csv",
            "train_frac": 0.8,
            "valid_frac": 0.2,
            "seed": 42,
            "fraction_data": 1,
            "batch_size": 64,
        },
    }

    from omegaconf import DictConfig

    config = DictConfig(config)
    train_molbind(config)
