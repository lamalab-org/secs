from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from molbind.models.model import MolBind
import torch
from torch.nn.functional import cosine_similarity
import pytorch_lightning as L
import hydra
from molbind.data.dataloaders import load_combined_loader, MODALITY_DATA_TYPES
import polars as pl


class MolBindModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.config = cfg
        self.loss = InfoNCE(temperature=cfg.loss.temperature, negative_mode="unpaired")

    def forward(self, input):
        return self.model(input)

    def _info_nce_loss(self, z1, z2):
        return self.loss(z1, z2)

    def _multimodal_loss(self, embeddings_dict, prefix):
        modality_pair = [*embeddings_dict.keys()]
        loss = self._info_nce_loss(
            embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]]
        )
        self.log(f"{prefix}_loss", loss)
        return loss

    def training_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict, "train")

    def validation_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )


def train_molbind(config: dict = None):
    wandb_logger = L.loggers.WandbLogger(
        project=config.wandb.project_name, entity=config.wandb.entity
    )

    device_count = torch.cuda.device_count()
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="cuda",
        log_every_n_steps=10,
        logger=wandb_logger,
        devices=device_count if device_count > 1 else "auto",
        strategy="ddp" if device_count > 1 else "auto",
    )

    train_modality_data = {}
    valid_modality_data = {}

    # Example SMILES - SELFIES modality pair:
    data = pl.read_csv(config.data.dataset_path)[:4096]
    shuffled_data = data.sample(fraction=1, shuffle=True, seed=42)
    dataset_length = len(shuffled_data)
    valid_shuffled_data = shuffled_data.tail(
        int(config.data.valid_frac * dataset_length)
    )
    train_shuffled_data = shuffled_data.head(
        int(config.data.train_frac * dataset_length)
    )

    columns = shuffled_data.columns
    # extract non-central modalities (i.e. not the central modality smiles)
    non_central_modalities = config.data.modalities

    for column in columns:
        if column in non_central_modalities:
            # drop nan for specific pair
            train_modality_smiles_pair = train_shuffled_data[
                ["smiles", column]
            ].drop_nulls()
            valid_modality_smiles_pair = valid_shuffled_data[
                ["smiles", column]
            ].drop_nulls()

            train_modality_data[column] = [
                train_modality_smiles_pair["smiles"].to_list(),
                train_modality_smiles_pair[column].to_list(),
            ]
            valid_modality_data[column] = [
                valid_modality_smiles_pair["smiles"].to_list(),
                valid_modality_smiles_pair[column].to_list(),
            ]

    combined_loader = load_combined_loader(
        data_modalities=train_modality_data,
        batch_size=256,
        shuffle=True,
        num_workers=1,
    )

    valid_dataloader = load_combined_loader(
        data_modalities=valid_modality_data,
        batch_size=256,
        shuffle=False,
        num_workers=1,
    )

    trainer.fit(
        MolBindModule(config),
        train_dataloaders=combined_loader,
        val_dataloaders=valid_dataloader,
    )


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
            "dataset_path": "selfies_smiles_data.csv",
            "train_frac": 0.8,
            "valid_frac": 0.2,
        },
    }

    from omegaconf import DictConfig

    config = DictConfig(config)
    train_molbind(config)
