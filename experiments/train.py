import datetime  # noqa: I002
import os
import re
from pathlib import Path

import hydra
import pytorch_lightning as L
import rootutils
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.strategies.ddp import DDPStrategy

from molbind.data.datamodule import MolBindDataModule
from molbind.data.molbind_dataset import MolBindDataset
from molbind.data.utils.file_utils import csv_load_function, pickle_load_function
from molbind.models.lightning_module import MolBindModule

load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

TRAIN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def train_molbind(config: DictConfig):
    wandb_logger = L.loggers.WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        id=config.run_id + "_" + TRAIN_DATE
        if hasattr(config, "run_id")
        else TRAIN_DATE,
    )

    world_size = torch.cuda.device_count()
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
        ).build_datasets_for_modalities(),
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=valid_shuffled_data,
            context_length=config.data.context_length,
        ).build_datasets_for_modalities(),
    )

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

    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger,
        num_nodes=config.trainer.num_nodes,
        devices=world_size if world_size > 1 else "auto",
        strategy=DDPStrategy(find_unused_parameters=True) if world_size > 1 else "auto",
        gradient_clip_val=0.5,
        precision=config.trainer.precision,
        deterministic=True,
    )

    trainer.fit(
        model=MolBindModule(config),
        datamodule=datamodule,
    )


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


@hydra.main(version_base="1.3", config_path="../configs")
def main(config: DictConfig):
    init_distributed_mode(12354)
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
    patch_lightning_slurm_master_addr()
    seed_everything(42)
    main()