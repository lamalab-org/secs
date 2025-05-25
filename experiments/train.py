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

from molbind.data.datamodule import MolBindDataModule
from molbind.data.molbind_dataset import MolBindDataset
from molbind.models.lightning_module import MolBindModule

load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

TRAIN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def train_molbind(config: DictConfig):
    # define the run_id based on the config name and the date
    run_id = config.run_id + "_" + TRAIN_DATE if hasattr(config, "run_id") else TRAIN_DATE
    # set wandb mode to offline if no WANDB_API_KEY is set
    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
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
    data = load_dataset(config.data.dataset_path)
    features = [*config.data.modalities, config.data.central_modality]
    train_data = data["train"].to_pandas()[features]
    logger.info(f"Train data shape: {train_data.shape}")
    valid_data = data["val"].to_pandas()[features]
    logger.info(f"Validation data shape: {valid_data.shape}")
    # Shuffling the data with a specified fraction and seed
    train_shuffled_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    valid_shuffled_data = valid_data.copy()

    # set up the dataloaders
    train_dataloader, valid_dataloader = (
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=train_shuffled_data,
            context_length=config.data.context_length,
        ).build_datasets_for_modalities(),
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=valid_shuffled_data,
            context_length=config.data.context_length,
        ).build_datasets_for_modalities(),
    )
    # set up the data module
    datamodule = MolBindDataModule(
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
    )
    # train the model
    trainer.fit(
        model=MolBindModule(config),
        datamodule=datamodule,
    )
    # copy the best model under the name "best_model"
    best_model_path = Path(config.callbacks.model_checkpoint.dirpath) / Path(run_id) / "best_model.ckpt"
    os.system(f"cp {best_model_path} {Path(config.callbacks.model_checkpoint.dirpath) / Path(run_id) / 'best_model.ckpt'}")
    logger.info(f"Best model saved at {best_model_path}")
    logger.info("Training complete")
    logger.info("Exiting")


def _get_first_node():
    """Return the first node we can find in the Slurm node list."""
    nodelist = os.getenv("SLURM_JOB_NODELIST")

    bracket_re = re.compile(r"(.*?)\[(.*?)\]")
    dash_re = re.compile("(.*?)-")
    comma_re = re.compile("(.*?),")

    bracket_result = bracket_re.match(nodelist)

    if bracket_result:
        node = bracket_result[1]
        indices = bracket_result[2]

        comma_result = comma_re.match(indices)
        if comma_result:
            indices = comma_result[1]

        dash_result = dash_re.match(indices)
        first_index = dash_result[1] if dash_result else indices

        return node + first_index

    comma_result = comma_re.match(nodelist)
    if comma_result:
        return comma_result[1]

    return nodelist


def init_distributed_mode(port=12354):
    """Initialize some environment variables for PyTorch Distributed
    using Slurm.
    """
    # The number of total processes started by Slurm.
    os.environ["WORLD_SIZE"] = os.getenv("SLURM_NTASKS")
    # Index of the current process.
    os.environ["RANK"] = os.getenv("SLURM_PROCID")
    # Index of the current process on this node only.
    os.environ["LOCAL_RANK"] = os.getenv("SLURM_LOCALID")

    master_addr = _get_first_node()
    systemname = os.getenv("SYSTEMNAME", "")
    # Need to append "i" on Jülich machines to connect across InfiniBand cells.
    if systemname in ["juwels", "juwelsbooster", "jureca"]:
        master_addr = master_addr + "i"
    os.environ["MASTER_ADDR"] = master_addr

    # An arbitrary free port on node 0.
    os.environ["MASTER_PORT"] = str(port)
    # print the environment variables
    logger.info(f"MASTER_ADDR={os.getenv('MASTER_ADDR')}")
    logger.info(f"MASTER_PORT={os.getenv('MASTER_PORT')}")
    logger.info(f"WORLD_SIZE={os.getenv('WORLD_SIZE')}")
    logger.info(f"RANK={os.getenv('RANK')}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="molbind_config.yaml")
def main(config: DictConfig):
    # init_distributed_mode(12354)
    torch.use_deterministic_algorithms(True, warn_only=True)
    train_molbind(config)


if __name__ == "__main__":
    seed_everything(42, workers=True)
    main()
