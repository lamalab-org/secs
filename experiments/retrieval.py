import datetime  # noqa: I002
import os
import pickle as pkl
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

load_dotenv(".env")
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# set an unique identifier for the retrieval run
RETRIEVAL_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def train_molbind(config: DictConfig):
    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        id="retrieval_" + RETRIEVAL_DATE,
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

    valid_datasets = (
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=valid_shuffled_data,
            context_length=config.data.context_length,
        ).build_datasets_for_modalities(),
    )
    valid_shuffled_data[[config.data.central_modality]].write_csv(
        f"{config.store_embeddings_directory}.csv"
    )

    datamodule = MolBindDataModule(
        data={
            "predict": valid_datasets,
            "dataloader_arguments": {
                "batch_size": config.data.batch_size,
                "num_workers": config.data.num_workers,
            },
        },
    )

    predictions = trainer.predict(
        model=MolBindModule(config),
        datamodule=datamodule,
    )
    # concatenate predictions outside of this script

    with open(f"{config.store_embeddings_directory}.pkl", "wb") as f:
        pkl.dump(predictions, f, protocol=pkl.HIGHEST_PROTOCOL)


@hydra.main(version_base="1.3", config_path="../configs")
def main(config: DictConfig):
    train_molbind(config)


if __name__ == "__main__":
    main()
