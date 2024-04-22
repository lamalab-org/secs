import os  # noqa: I002
import pickle as pkl
from pathlib import Path

import hydra
import polars as pl
import pytorch_lightning as L
import rootutils
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from molbind.data.datamodule import MolBindDataModule
from molbind.data.molbind_dataset import MolBindDataset
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

    # extract format of dataset file
    data_format = Path(config.data.dataset_path).suffix
    # check if dataset is in csv or pkl format
    if data_format == ".csv":
        data = pl.read_csv(config.data.dataset_path)
    elif data_format == ".pkl":
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
