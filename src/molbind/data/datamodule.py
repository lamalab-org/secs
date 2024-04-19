from typing import Dict, Optional  # noqa: UP035, I002

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from pytorch_lightning import LightningDataModule


class MolBindDataModule(LightningDataModule):
    def __init__(
        self,
        data: Dict,  # noqa: UP006
    ) -> None:
        super().__init__()
        self.train_dataloaders = data["train"]
        self.val_dataloaders = data["val"]

    def setup(self, stage: Optional[str] = None) -> None:
        # for now the train/val data loaders are already set up
        pass

    def train_dataloader(self) -> CombinedLoader:
        # iter through train data loaders
        return self.train_dataloaders

    def val_dataloader(self) -> CombinedLoader:
        # iter through val data loaders
        return self.val_dataloaders
