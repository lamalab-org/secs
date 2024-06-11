import datetime  # noqa: I002
import os
from glob import glob

import hydra
import rootutils
from dotenv import load_dotenv
from lightning import Trainer
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molbind.data.components.datasets import StructureDataset
from molbind.models.components.dimenet_lightningmodule import StructureEncoderModule

load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def train_dimenet(config: DictConfig):
    wandb_logger = WandbLogger(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        id="dimenet_" + datetime.datetime.now().strftime("%Y%m%d_%H%M"),
    )
    CHEMBL_100k = glob(  # noqa: PTH207
        f"{config.data.directory_with_structures}/*/conf_00.sdf"
    )[:100000]
    train_files = CHEMBL_100k[: int(len(CHEMBL_100k) * 0.8)]
    val_files = CHEMBL_100k[int(len(CHEMBL_100k) * 0.8) :]
    val_dataset = StructureDataset(sdf_file_list=val_files, dataset_mode="encoder")

    train_dataset = StructureDataset(sdf_file_list=train_files, dataset_mode="encoder")

    train_struct_dataloader = GeometricDataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor=2,
        num_workers=2,
    )

    valid_struct_dataloader = GeometricDataLoader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        drop_last=True,
        prefetch_factor=2,
        num_workers=2,
    )

    trainer = Trainer(
        max_epochs=100,
        devices=1,
        precision="16-mixed",
        deterministic=True,
        accelerator="cuda",
        logger=wandb_logger,
    )

    trainer.fit(
        StructureEncoderModule(config),
        train_dataloaders=train_struct_dataloader,
        val_dataloaders=valid_struct_dataloader,
    )


@hydra.main(version_base="1.3", config_path="../configs")
def main(config: DictConfig) -> None:
    train_dimenet(config)


if __name__ == "__main__":
    main()
