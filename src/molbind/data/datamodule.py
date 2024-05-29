from typing import Dict  # noqa: UP035, I002

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, DistributedSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader

from molbind.data.available import NonStringModalities


class MolBindDataModule(LightningDataModule):
    def __init__(
        self,
        data: Dict,  # noqa: UP006
    ) -> None:
        super().__init__()
        # create attributes for each subset
        # and add dataloader arguments
        self.datasets = {}
        for subset in ["train", "val", "test"]:
            if subset in data:
                self.datasets[subset] = data[subset]
        self.dataloader_arguments = data["dataloader_arguments"]

    def build_multimodal_dataloader(
        self,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        num_workers: int,
        mode: str,
    ) -> CombinedLoader:
        dataloaders = {}
        for modality in self.datasets[mode]:
            if modality == NonStringModalities.GRAPH:
                dataloaders[modality] = GeometricDataLoader(
                    self.datasets[mode][modality],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    sampler=DistributedSampler(
                        self.datasets[mode][modality], shuffle=shuffle
                    ),
                )
            else:
                dataloaders[modality] = DataLoader(
                    self.datasets[mode][modality],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    sampler=DistributedSampler(
                        self.datasets[mode][modality], shuffle=shuffle
                    ),
                )
        # CombinedLoader does not work with DDPSampler directly
        # So each dataloader has a DistributedSampler
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
            drop_last=True,
            shuffle=False,
            num_workers=self.dataloader_arguments["num_workers"],
            mode="val",
        )
        return CombinedLoader(val_dataloaders, "sequential")

    def test_dataloader(self) -> CombinedLoader:
        # iter through test data loaders
        return self.test_dataloaders if hasattr(self, "test_dataloaders") else None
