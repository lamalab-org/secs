import datetime  # noqa: I002
import os
from pathlib import Path

import hydra
import pytorch_lightning as L
import rootutils
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from molbind.data.datamodule import MolBindDataModule
from molbind.data.molbind_dataset import MolBindDataset
from molbind.data.utils.file_utils import csv_load_function, pickle_load_function
from molbind.models.lightning_module import MolBindModule

TRAIN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")

load_dotenv(".env")

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def train_molbind(config: DictConfig):
    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        id=config.run_id + "_" + TRAIN_DATE
        if hasattr(config, "run_id")
        else TRAIN_DATE,
    )

    device_count = torch.cuda.device_count()
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger,
        devices=device_count if device_count > 1 else "auto",
        strategy="ddp" if device_count > 1 else "auto",
        precision=config.trainer.precision,
    )

    # extract format of dataset file
    data_format = Path(config.data.dataset_path).suffix

    handlers = {
        ".csv": csv_load_function,
        ".pickle": pickle_load_function,
        ".pkl": pickle_load_function,
    }

    try:
        data = handlers[data_format](config.data.dataset_path)
    except KeyError:
        logger.error(f"Format {data_format} not supported")

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


@hydra.main(version_base="1.3", config_path="../configs")
def main(config: DictConfig):
    train_molbind(config)


if __name__ == "__main__":
    main()
