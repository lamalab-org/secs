import os  # noqa: I002
import pickle as pkl

import hydra
import polars as pl
import pytorch_lightning as L
import rootutils
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from molbind.data.dataloaders import load_combined_loader
from molbind.data.datamodule import MolBindDataModule
from molbind.models.lightning_module import MolBindModule

load_dotenv(".env")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def train_molbind(config: DictConfig):
    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
    )

    device_count = torch.cuda.device_count()
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger,
        devices=device_count if device_count > 1 else "auto",
        strategy="ddp" if device_count > 1 else "auto",
    )

    train_modality_data = {}
    valid_modality_data = {}

    # check format of config.data.dataset_path
    data_format = config.data.dataset_path.split(".")[-1]

    if data_format == "csv":
        data = pl.read_csv(config.data.dataset_path)
    elif data_format == "pkl":
        data = pkl.load(open(config.data.dataset_path, "rb"))  # noqa: PTH123, SIM115
        data = pl.from_pandas(data)
    shuffled_data = data.sample(
        fraction=config.data.fraction_data, shuffle=True, seed=config.data.seed
    )
    dataset_length = len(shuffled_data)
    valid_shuffled_data = shuffled_data.tail(
        int(config.data.valid_frac * dataset_length)
    )
    train_shuffled_data = shuffled_data.head(
        int(config.data.train_frac * dataset_length)
    )

    columns = shuffled_data.columns
    # extract non-central modalities
    non_central_modalities = config.data.modalities
    central_modality = config.data.central_modality

    # for column in columns:
    #     if column in non_central_modalities:
    #         # drop nan for specific pair
    #         train_modality_pair = train_shuffled_data[
    #             [central_modality, column]
    #         ].drop_nulls()
    #         valid_modality_pair = valid_shuffled_data[
    #             [central_modality, column]
    #         ].drop_nulls()

    #         train_modality_data[column] = [
    #             train_modality_pair[central_modality].to_list(),
    #             train_modality_pair[column].to_list(),
    #         ]
    #         valid_modality_data[column] = [
    #             valid_modality_pair[central_modality].to_list(),
    #             valid_modality_pair[column].to_list(),
    #         ]

    from molbind.data.molbind_dataset import MolBindDataset

    train_dataloader, valid_dataloader = (
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=train_shuffled_data,
            context_length=config.data.context_length,
        ).build_multimodal_dataloader(
            config.data.batch_size,
            shuffle=True,
            drop_last=config.data.drop_last,
            num_workers=config.data.num_workers,
        ),
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=valid_shuffled_data,
            context_length=config.data.context_length,
        ).build_multimodal_dataloader(
            config.data.batch_size,
            shuffle=False,
            drop_last=config.data.drop_last,
            num_workers=config.data.num_workers,
        ),
    )

    # train_dataloader = load_combined_loader(
    #     central_modality=config.data.central_modality,
    #     data_modalities=train_modality_data,
    #     batch_size=config.data.batch_size,
    #     shuffle=True,
    #     num_workers=1,
    # )

    # valid_dataloader = load_combined_loader(
    #     central_modality=config.data.central_modality,
    #     data_modalities=valid_modality_data,
    #     batch_size=config.data.batch_size,
    #     shuffle=False,
    #     num_workers=1,
    # )

    datamodule = MolBindDataModule(
        data={"train": train_dataloader, "val": valid_dataloader},
    )

    trainer.fit(
        model=MolBindModule(config),
        datamodule=datamodule,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(config: DictConfig):
    train_molbind(config)


if __name__ == "__main__":
    main()
