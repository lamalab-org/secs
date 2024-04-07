from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from molbind.models.model import MolBind
import torch
from torch.nn.functional import cosine_similarity
import pytorch_lightning as L
import hydra


class MolBindModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.config = cfg
        self.loss = InfoNCE(temperature=cfg.loss.temperature, negative_mode="unpaired")

    def forward(self, input):
        return self.model(input)

    def _info_nce_loss(self, z1, z2):
        return self.loss(z1, z2)

    def _multimodal_loss(self, embeddings_dict, prefix):
        modality_pair = [*embeddings_dict.keys()]
        loss = self._info_nce_loss(embeddings_dict[modality_pair[0]], embeddings_dict[modality_pair[1]])
        self.log(f"{prefix}_loss", loss)
        return loss
    
    def training_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict, "train")
    
    def validation_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict, "val")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)


from molbind.data.dataloaders import load_combined_loader
import polars as pl


def train_molbind(config : dict = None):
    
    wandb_logger = L.loggers.WandbLogger(project="embedbind", entity="adrianmirza")

    device_count = torch.cuda.device_count()
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="cuda",
        log_every_n_steps=30,
        logger=wandb_logger,
        devices= device_count if device_count > 1 else "auto",
        strategy="ddp" if device_count > 1 else "auto"
    )

    # Example SMILES - SELFIES modality pair:
    data = pl.read_csv("selfies_smiles_data.csv")[:1000]
    shuffled_selfies = data.sample(fraction=1, shuffle=True, seed=42)
    train_selfies = shuffled_selfies.head(int(0.8*len(shuffled_selfies)))
    valid_selfies = shuffled_selfies.tail(int(0.2*len(shuffled_selfies)))
    
    combined_loader = load_combined_loader(
        data_modalities = {
        "selfies" : [train_selfies["smiles"].to_list(), train_selfies["selfies"].to_list()],
        },
        batch_size=100,
        shuffle=False,
        num_workers=1)

    config = {
        "model" : {"projection_heads" : {"selfies" : {"dims" : [256, 128]}, "smiles" : {"dims" : [256, 128]}}},
        "loss" : {"temperature" : 0.1},
        "optimizer" : {"lr" : 1e-4, "weight_decay" : 1e-4},
        "data" : {"modalities" : ["selfies"]},
    }
    
    from omegaconf import DictConfig
    config = DictConfig(config)
    
    trainer.fit(MolBindModule(config), combined_loader)

# train model with Trainer

if __name__ == "__main__":
    train_molbind()
