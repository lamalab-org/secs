import datetime
import os
import re
from pathlib import Path
import gc

import hydra
import pandas as pd
import pytorch_lightning as L
import rootutils
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.environments import SLURMEnvironment
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

    # Memory optimization environment variables
    try:
        # set PYTORCH_ALLOC_CONF to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        logger.warning("Your PyTorch version does not support PYTORCH_CUDA_ALLOC_CONF")

    # Additional memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        id=run_id,
    )
    wandb_logger.log_hyperparams(config)

    # define the number of GPUs available for the dataloaders
    world_size = torch.cuda.device_count()

    # extract format of dataset file
    data_format = Path(config.data.dataset_path).suffix
    handlers = {
        ".csv": pd.read_csv,
        ".pickle": pd.read_pickle,
        ".pkl": pd.read_pickle,
        ".parquet": pd.read_parquet,
    }

    # load and handle the data with memory optimization
    try:
        logger.info(f"Loading data from {config.data.dataset_path}")
        data = handlers[data_format](config.data.dataset_path)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
    except KeyError:
        logger.error(f"Format {data_format} not supported")
        raise

    # Shuffling the data with a specified fraction and seed
    shuffled_data = data.sample(frac=config.data.fraction_data, random_state=config.data.seed)

    # Clear original data from memory
    del data
    gc.collect()

    # Get the total length of the dataset
    dataset_length = len(shuffled_data)

    # Split the data into validation and training datasets
    valid_shuffled_data = shuffled_data.tail(int(config.data.valid_frac * dataset_length)).copy()
    train_shuffled_data = shuffled_data.head(int(config.data.train_frac * dataset_length)).copy()

    # Clear shuffled_data from memory
    del shuffled_data
    gc.collect()

    # Log dataset sizes
    logger.info(f"Total samples after shuffling: {dataset_length}")
    logger.info(f"Training samples: {len(train_shuffled_data)}")
    logger.info(f"Validation samples: {len(valid_shuffled_data)}")

    # set up the dataloaders with memory optimization
    logger.info("Creating training dataset...")
    train_dataset = MolBindDataset(
        central_modality=config.data.central_modality,
        other_modalities=config.data.modalities,
        data=train_shuffled_data,
        context_length=config.data.context_length,
    )
    train_dataloader = train_dataset.build_datasets_for_modalities()

    # Clear training data from memory after processing
    del train_shuffled_data, train_dataset
    gc.collect()

    logger.info("Creating validation dataset...")
    valid_dataset = MolBindDataset(
        central_modality=config.data.central_modality,
        other_modalities=config.data.modalities,
        data=valid_shuffled_data,
        context_length=config.data.context_length,
    )
    valid_dataloader = valid_dataset.build_datasets_for_modalities()

    # Clear validation data from memory after processing
    del valid_shuffled_data, valid_dataset
    gc.collect()

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

    # Memory-optimized trainer configuration
    trainer_config = {
        "max_epochs": config.trainer.max_epochs,
        "accelerator": config.trainer.accelerator,
        "log_every_n_steps": config.trainer.log_every_n_steps,
        "logger": wandb_logger,
        "callbacks": callbacks,
        "num_nodes": config.trainer.num_nodes,
        "gradient_clip_val": 0.5,
        "precision": config.trainer.precision,
        "deterministic": True,
    }

    # Configure distributed training with memory optimizations
    if world_size > 1:
        trainer_config.update(
            {
                "devices": world_size,
                "strategy": DDPStrategy(
                    find_unused_parameters=True,  # Set to True for multimodal models with changing graphs
                    gradient_as_bucket_view=True,  # Memory optimization
                    static_graph=False,  # Set to False for multimodal models with dynamic graphs
                ),
            }
        )
        logger.info(f"Using distributed training with {world_size} GPUs")
    else:
        trainer_config.update(
            {
                "devices": "auto",
                "strategy": "auto",
            }
        )
        logger.info("Using single GPU training")

    # set up the trainer
    trainer = L.Trainer(**trainer_config)

    # Initialize model with memory optimization
    logger.info("Initializing model...")
    model = MolBindModule(config)

    # train the model
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

    # copy the best model under the name "best_model"
    best_model_path = Path(config.callbacks.model_checkpoint.dirpath) / Path(run_id) / "best_model.ckpt"
    try:
        # Find the actual best checkpoint
        checkpoint_dir = Path(config.callbacks.model_checkpoint.dirpath) / Path(run_id)
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
            if checkpoint_files:
                # Get the most recent checkpoint (or implement your own logic)
                best_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
                os.system(f"cp {best_checkpoint} {best_model_path}")
                logger.info(f"Best model saved at {best_model_path}")
            else:
                logger.warning("No checkpoint files found")
        else:
            logger.warning("Checkpoint directory does not exist")
    except Exception as e:
        logger.error(f"Error copying best model: {e}")

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
    train_molbind(config)


def patch_lightning_slurm_master_addr():
    # Quit if we're not on a Jülich machine.
    if os.getenv("SYSTEMNAME", "") not in [
        "juwelsbooster",
        "juwels",
        "jurecadc",
    ]:
        return

    old_resolver = SLURMEnvironment.resolve_root_node_address

    def new_resolver(nodes):
        # Append an i" for communication over InfiniBand.
        return old_resolver(nodes) + "i"

    SLURMEnvironment.__old_resolve_root_node_address = old_resolver
    SLURMEnvironment.resolve_root_node_address = new_resolver


if __name__ == "__main__":
    # patch_lightning_slurm_master_addr()
    seed_everything(42)
    main()
