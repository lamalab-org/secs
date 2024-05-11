from typing import Dict, Optional  # noqa: UP035, I002

from lightning.pytorch.utilities.combined_loader import CombinedLoader
from pytorch_lightning import LightningDataModule


class MolBindDataModule(LightningDataModule):
    def __init__(
        self,
        data: Dict,  # noqa: UP006
    ) -> None:
        super().__init__()
        for subset in ["train", "val", "test"]:
            if subset in [*data]:
                setattr(self, f"{subset}_dataloaders", data[subset])

    def setup(self, stage: Optional[str] = None) -> None:
        # for now the train/val data loaders are already set up
        pass

    def train_dataloader(self) -> CombinedLoader:
        # iter through train data loaders
        return self.train_dataloaders

    def val_dataloader(self) -> CombinedLoader:
        # iter through val data loaders
        return self.val_dataloaders if hasattr(self, "val_dataloaders") else None

    def test_dataloader(self) -> CombinedLoader:
        # iter through test data loaders
        return self.test_dataloaders if hasattr(self, "test_dataloaders") else None
