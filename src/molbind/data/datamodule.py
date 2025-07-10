from typing import Literal

import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from loguru import logger
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler

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
                    drop_last=drop_last,
                )
                shuffle = False
            else:
                distributed_sampler = None

            # Optimize memory usage for distributed training
            if self.distributed:
                prefetch_factor = 1
                actual_num_workers = min(num_workers, 2)
                # Disable persistent workers in distributed training to save memory
                persistent_workers = False
                # Disable pin_memory in distributed training to save GPU memory
                pin_memory = False
            else:
                prefetch_factor = min(2, num_workers) if num_workers > 0 else 2
                actual_num_workers = num_workers
                persistent_workers = num_workers > 0
                pin_memory = True

            dataloaders[modality] = DataLoader(
                self.datasets[mode][modality],
                batch_size=batch_size,
                num_workers=actual_num_workers,
                drop_last=drop_last,
                sampler=distributed_sampler,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
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
        val_dataloaders = self.build_multimodal_dataloader(
            batch_size=self.dataloader_arguments["batch_size"],
            drop_last=True,
            shuffle=False,
            num_workers=self.dataloader_arguments["num_workers"],
            mode="val",
        )
        return CombinedLoader(val_dataloaders, "sequential")

    def predict_dataloader(self) -> CombinedLoader:
        # iter through test data loaders
        test_dataloaders = self.build_predict_dataloader(
            batch_size=self.dataloader_arguments["batch_size"],
            drop_last=False,
            shuffle=False,
            num_workers=self.dataloader_arguments["num_workers"],
            mode="predict",
        )
        return CombinedLoader(test_dataloaders, "sequential")

    def build_predict_dataloader(
        self,
        batch_size: int | dict[str, int],
        drop_last: bool,
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
        # Use minimal prefetch for prediction to save memory
        prefetch_factor = 1
        # Reduce workers for prediction
        actual_num_workers = min(num_workers, 2)

        for modality in self.datasets[mode][0]:
            dataloaders[modality] = DataLoader(
                self.datasets[mode][0][modality],
                batch_size=batch_size,
                num_workers=actual_num_workers,
                drop_last=False,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
                pin_memory=False,
                persistent_workers=False,
            )
        return dataloaders

    def embed_dataloader(self, tokenized_data: list[list[int]]) -> StringDatasetEmbedding:
        dataset = StringDatasetEmbedding(tokenized_data)

        embedding_num_workers = min(self.dataloader_arguments["num_workers"], 2)

        return DataLoader(
            dataset,
            batch_size=self.dataloader_arguments["batch_size"],
            num_workers=embedding_num_workers,
            drop_last=False,
            shuffle=False,
            prefetch_factor=1,  # Minimal prefetch
            pin_memory=False,
            persistent_workers=False,
        )
