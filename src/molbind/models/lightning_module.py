from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from molbind.models.model import MolBind
import torch
from torch.nn.functional import cosine_similarity


class MolBindModule(LightningModule):
    def __init__(self, cfg)
        
        super().__init__()
        self.model = MolBind(cfg=cfg)
        self.config = cfg
        
    def forward(self, input):
        embedding_dict = self.model(input)
        return embedding_dict

    def _info_nce_loss(self, z1, z2, temperature, prefix):
        loss = InfoNCE(temperature=temperature, negative_mode="unpaired")
        loss = loss(z1, z2)
        print(f"{prefix}_loss", loss)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_cosine_similarity", cosine_similarity(z1, z2, dim=1).mean())
        return loss
    
    def _multimodal_loss(embeddings_dict):
        loss = 0
        for modality_embedding in embeddings_dict.keys():
            # loss as sum of pairs of embeddings (smiles-modality_1, smiles-modality_2)
            if modality_embedding != "smiles":
                loss += self._info_nce_loss(embeddings_dict["smiles"], embeddings_dict[modality_embedding], temperature=0.1, prefix=modality_embedding)
        return loss
    
    def training_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict)
    
    def validation_step(self, input):
        embeddings_dict = self.forward(input)
        return self._multimodal_loss(embeddings_dict)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.optimizer.lr, weight_decay=self.config.optimizer.weight_decay)