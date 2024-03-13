from molbind.models.components.custom_encoders import SmilesEncoder, SelfiesEncoder
from molbind.models.components.head import ProjectionHead
from molbind.models.component.pooler import EmbedPooler
from molbind.data.dataloaders import MolBindDataModule
import pytorch_lightning as pl


class MolBind(nn.Module):
    def __init__(self, model_opt_params):
        super(MolBind, self).__init__()
        self.smiles_encoder = SmilesEncoder()
        self.selfies_encoder = SelfiesEncoder()
        self.projection_head = ProjectionHead()
        self.pooler = EmbedPooler()
        
    def forward(self, input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2):
        smiles_embedding = self.smiles_encoder(input_ids_modality_1, attention_mask_modality_1)
        selfies_embedding = self.selfies_encoder(input_ids_modality_2, attention_mask_modality_2)
        joint_embedding = self.projection_head(smiles_embedding, selfies_embedding)
        return joint_embedding
        
