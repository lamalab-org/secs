from typing import Literal

import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molbind.data.available import NonStringModalities
from molbind.data.components.datasets import StringDatasetEmbedding


class MolBindDataModule(LightningDataModule):
    def __init__(
        self,
        data: dict,
    ) -> None:
        super().__init__()
        # create attributes for each subset
        # and add dataloader arguments
        self.datasets = {}
        for subset in ["train", "val", "test", "predict"]:
            if subset in data:
                self.datasets[subset] = data[subset]
        if "dataloader_arguments" in data:
            self.dataloader_arguments = data["dataloader_arguments"]

        self.distributed = torch.cuda.device_count() > 1

    def build_multimodal_dataloader(
        self,
        mode: Literal["train", "val", "test"],
        batch_size: int | dict[str, int],
        drop_last: bool,
        shuffle: bool,
        num_workers: int = 2,
    ) -> CombinedLoader:
        dataloaders = {}

        for modality in self.datasets[mode]:
            if self.distributed:
                distributed_sampler = DistributedSampler(
                    self.datasets[mode][modality],
                    shuffle=shuffle,
                )
                shuffle = None
            else:
                distributed_sampler = None
            if modality == NonStringModalities.GRAPH or modality == NonStringModalities.STRUCTURE:
                dataloaders[modality] = GeometricDataLoader(
                    self.datasets[mode][modality],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    sampler=distributed_sampler,
                    shuffle=shuffle,
                    prefetch_factor=num_workers,
                    persistent_workers=True,
                )
            else:
                dataloaders[modality] = DataLoader(
                    self.datasets[mode][modality],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    sampler=distributed_sampler,
                    shuffle=shuffle,
                    prefetch_factor=num_workers,
                    persistent_workers=True,
                )
        # CombinedLoader does not work with DDPSampler directly
        # So each dataloader has a DistributedSampler
        logger.info(f"Nr of dataloaders: {len(dataloaders)}")
        return dataloaders

    def train_dataloader(self) -> CombinedLoader:
        # iter through train data loaders
        # add DDPSampler to train_dataloader
        train_dataloaders = self.build_multimodal_dataloader(
            batch_size=self.dataloader_arguments["batch_size"],
            drop_last=True,
            shuffle=True,
            num_workers=self.dataloader_arguments["num_workers"],
            mode="train",
        )
        return CombinedLoader(train_dataloaders, "sequential")

    def val_dataloader(self) -> CombinedLoader:
        # iter through val data loaders
        # add DDPSampler to val_dataloader
        # for val_dataloader in [*self.val_dataloaders.values()]:
        #     val_dataloader.sampler = DistributedSampler(val_dataloader.dataset)
        val_dataloaders = self.build_multimodal_dataloader(
            batch_size=self.dataloader_arguments["batch_size"],
            drop_last=False,
            shuffle=False,
            num_workers=self.dataloader_arguments["num_workers"],
            mode="val",
        )
        return CombinedLoader(val_dataloaders, "sequential")

    def predict_dataloader(self) -> CombinedLoader:
        # iter through test data loaders
        test_dataloaders = self.build_predict_dataloader(
            batch_size=self.dataloader_arguments["batch_size"],
            shuffle=False,
            num_workers=self.dataloader_arguments["num_workers"],
            mode="predict",
        )
        return CombinedLoader(test_dataloaders, "sequential")

    def build_predict_dataloader(
        self,
        batch_size: int | dict[str, int],
        shuffle: bool,
        num_workers: int,
        mode: str,
    ) -> dict[str, DataLoader]:
        """
        Build dataloaders for the predict step.
        This function is similar to `build_multimodal_dataloader` but does not use the DistributedSampler.
        Hence, it can be ran on a single GPU.

        After, in the `retrieval.py` script the predictions are concatenated.
        """
        dataloaders = {}
        for modality in self.datasets[mode][0]:
            if modality == NonStringModalities.GRAPH or modality == NonStringModalities.STRUCTURE:
                dataloaders[modality] = GeometricDataLoader(
                    self.datasets[mode][0][modality],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=False,
                    shuffle=shuffle,
                    prefetch_factor=num_workers,
                )
            else:
                dataloaders[modality] = DataLoader(
                    self.datasets[mode][0][modality],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=False,
                    shuffle=shuffle,
                    prefetch_factor=num_workers,
                )
        # CombinedLoader does not work with DDPSampler directly
        # So each dataloader has a DistributedSampler
        return dataloaders

    def embed_dataloader(self, tokenized_data: list[list[int]]) -> StringDatasetEmbedding:
        dataset = StringDatasetEmbedding(tokenized_data)
        return DataLoader(
            dataset,
            batch_size=self.dataloader_arguments["batch_size"],
            num_workers=self.dataloader_arguments["num_workers"],
            drop_last=False,
            shuffle=False,
            prefetch_factor=self.dataloader_arguments["num_workers"],
        )
