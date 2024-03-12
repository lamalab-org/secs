from info_nce import InfoNCE
from pytorch_lightning import LightningModule
from molbind.models.components import MolBind
import torch
from torch.nn.functional import cosine_similarity


class MolBindModule(LightningModule):
    def __init__(self, 
                model_opt_params = {
                    "modality_1": "smiles",
                    "modality_2": "selfies",
                    "learning_rate" : 1e-4,
                    "temperature" : 0.06,
                    "modality_1_fc" : [768, 512],
                    "modality_2_fc" : [768, 512]
                    }):
        
        super().__init__()
        self.model = MolBind(model_opt_params)
        
        self.modality_1 = "smiles"
        self.modality_2 = model_opt_params["modality_2"]
        self.learning_rate = model_opt_params["learning_rate"]
        self.temperature = model_opt_params["temperature"]

        
    def forward(self, input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2):
        joint_embedding = self.model(input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2)
        return joint_embedding

    def _info_nce_loss(self, z1, z2, temperature, prefix):
        loss = InfoNCE(temperature=temperature, negative_mode="unpaired")
        loss = loss(z1, z2)
        print(f"{prefix}_loss", loss)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_cosine_similarity", cosine_similarity(z1, z2, dim=1).mean())
        return loss
    
    def training_step(self, batch, batch_idx):
        input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2 = batch
        selfies_embedding, smiles_embedding = self.forward(input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2)
        return self._info_nce_loss(smiles_embedding, selfies_embedding, self.temperature, prefix="train")

    def validation_step(self, batch, batch_idx):
        input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2 = batch

        selfies_embedding, smiles_embedding = self.model(input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2)
        return self._info_nce_loss(smiles_embedding, selfies_embedding, self.temperature, prefix="val")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)