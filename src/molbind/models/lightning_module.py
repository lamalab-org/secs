from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from molbind.models.model import MolBind
import torch
import pytorch_lightning as L
from molbind.data.dataloaders import load_combined_loader
from typing import Dict
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
            lr=self.config.model.optimizer.lr,
            weight_decay=self.config.model.optimizer.weight_decay,
        )


# def train_molbind(config: Dict = None):
#     wandb_logger = L.loggers.WandbLogger(
#         project=config.wandb.project_name, entity=config.wandb.entity
#     )

#     device_count = torch.cuda.device_count()
#     trainer = L.Trainer(
#         max_epochs=100,
#         accelerator="cuda",
#         log_every_n_steps=10,
#         logger=wandb_logger,
#         devices=device_count if device_count > 1 else "auto",
#         strategy="ddp" if device_count > 1 else "auto",
#     )

#     train_modality_data = {}
#     valid_modality_data = {}

#     data = pl.read_csv(config.data.dataset_path)
#     shuffled_data = data.sample(
#         fraction=config.data.fraction_data, shuffle=True, seed=config.data.seed
#     )
#     dataset_length = len(shuffled_data)
#     valid_shuffled_data = shuffled_data.tail(
#         int(config.data.valid_frac * dataset_length)
#     )
#     train_shuffled_data = shuffled_data.head(
#         int(config.data.train_frac * dataset_length)
#     )

#     columns = shuffled_data.columns
#     # extract non-central modalities (i.e. not the central modality smiles)
#     non_central_modalities = config.data.modalities
#     central_modality = config.data.central_modality

#     for column in columns:
#         if column in non_central_modalities:
#             # drop nan for specific pair
#             train_modality_pair = train_shuffled_data[
#                 [central_modality, column]
#             ].drop_nulls()
#             valid_modality_pair = valid_shuffled_data[
#                 [central_modality, column]
#             ].drop_nulls()

#             train_modality_data[column] = [
#                 train_modality_pair[central_modality].to_list(),
#                 train_modality_pair[column].to_list(),
#             ]
#             valid_modality_data[column] = [
#                 valid_modality_pair[central_modality].to_list(),
#                 valid_modality_pair[column].to_list(),
#             ]

#     combined_loader = load_combined_loader(
#         data_modalities=train_modality_data,
#         batch_size=config.data.batch_size,
#         shuffle=True,
#         num_workers=1,
#     )

#     valid_dataloader = load_combined_loader(
#         data_modalities=valid_modality_data,
#         batch_size=config.data.batch_size,
#         shuffle=False,
#         num_workers=1,
#     )

#     trainer.fit(
#         MolBindModule(config),
#         train_dataloaders=combined_loader,
#         val_dataloaders=valid_dataloader,
#     )
