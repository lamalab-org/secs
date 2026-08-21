import datetime
import os
import re
from pathlib import Path

import hydra
import pytorch_lightning as L
import rootutils
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

from secs.data.datamodule import SECSDataModule
from secs.data.secs_dataset import SECSDataset
from secs.models.lightning_module import SECSModule

load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

TRAIN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def train_molbind(config: DictConfig):
    # define the run_id based on the config name and the date
    run_id = config.run_id + "_" + TRAIN_DATE if hasattr(config, "run_id") else TRAIN_DATE

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        # set PYTORCH_ALLOC_CONF to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        logger.warning("Your PyTorch version does not support PYTORCH_CUDA_ALLOC_CONF")

    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        id=run_id,
    )
    # define the number of GPUs available for the dataloaders
    world_size = torch.cuda.device_count()
    # load and handle the data
    data = load_dataset(
        config.data.dataset_path, config.data.dataset_config if hasattr(config.data, "dataset_config") else "default"
    )
    features = [*config.data.modalities, config.data.central_modality]  # , "x_min", "x_max"]
    
    train_data = data["train"].to_pandas()[features]
    valid_data = data["val"].to_pandas()[features]
    
    logger.info(f"Train data shape: {train_data.shape}")
    logger.info(f"Validation data shape: {valid_data.shape}")

    train_shuffled_data = train_data.sort_values(by=config.data.central_modality).reset_index(drop=True)
    valid_shuffled_data = valid_data.copy()

    # set up the dataloaders
    train_dataloader, valid_dataloader = (
        SECSDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=train_shuffled_data,
            context_length=config.data.context_length,
            config=config,
        ).build_datasets_for_modalities(),
        SECSDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=valid_shuffled_data,
            context_length=config.data.context_length,
            config=config,
            split="val",
        ).build_datasets_for_modalities(),
    )
    # set up the data module
    datamodule = SECSDataModule(
        data={
            "train": train_dataloader,
            "val": valid_dataloader,
            "dataloader_arguments": {
                "batch_size": config.data.batch_size,
                "num_workers": config.data.num_workers,
            },
        },
    )
    # set up callbacks for the model
    callbacks = [
        ModelCheckpoint(
            monitor=config.callbacks.model_checkpoint.monitor,
            mode=config.callbacks.model_checkpoint.mode,
            save_top_k=config.callbacks.model_checkpoint.save_top_k,
            save_last=config.callbacks.model_checkpoint.save_last,
            filename=config.callbacks.model_checkpoint.filename,
            dirpath=Path(config.callbacks.model_checkpoint.dirpath) / Path(run_id),
        ),
        EarlyStopping(
            monitor=config.callbacks.early_stopping.monitor,
            mode=config.callbacks.early_stopping.mode,
            patience=config.callbacks.early_stopping.patience,
        ),
    ]

    # set up the trainer
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        num_nodes=config.trainer.num_nodes,
        devices=world_size if world_size > 1 else "auto",
        strategy=DDPStrategy(find_unused_parameters=True) if world_size > 1 else "auto",
        gradient_clip_val=2.0,
        precision=config.trainer.precision,
        deterministic=True,
        reload_dataloaders_every_n_epochs=1,
    )
    # train the model
    trainer.fit(
        model=SECSModule(config),
        datamodule=datamodule,
    )

    logger.info("Training complete")
    logger.info("Exiting")


@hydra.main(version_base="1.3", config_path="../configs", config_name="molbind_config.yaml")
def main(config: DictConfig):
    # init_distributed_mode(12354)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_molbind(config)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    main()
