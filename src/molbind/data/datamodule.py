from typing import Dict, List, Optional

from pytorch_lightning import LightningDataModule


class MolBindDataModule(LightningDataModule):
    def __init__(
        self,
        data: Dict,
        batch_size: int,
        central_modality: str,
        data_modalities: List[str],
    ):
        super().__init__()
        self.train_dataloaders = data["train"]
        self.val_dataloaders = data["val"]
        self.batch_size = batch_size
        self.central_modality = central_modality
        self.data_modalities = data_modalities

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return self.train_dataloaders

    def val_dataloader(self):
        return self.val_dataloaders
