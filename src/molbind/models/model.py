from molbind.models.components.custom_encoders import SmilesEncoder, SelfiesEncoder, GraphEncoder
from molbind.models.components.head import ProjectionHead
from molbind.models.component.pooler import EmbedPooler
from molbind.data.dataloaders import MolBindDataModule
import pytorch_lightning as pl
from torch import Tensor


class MolBind(nn.Module):
    def __init__(self, cfg):
        super(MolBind, self).__init__()
        # Instantiate all encoders
        self.smiles_encoder = SmilesEncoder(cfg.model.smiles_encoder)
        self.selfies_encoder = SelfiesEncoder(cfg.model.selfies_encoder)
        self.graph_encoder = GraphEncoder(cfg.model.graph_encoder)
        self.nmr_encoder = None
        
        self.dict_encoders = {
            "smiles" : self.smiles_encoder,
            "selfies" : self.selfies_encoder,
            "graph" : self.graph_encoder,
            "nmr" : self.nmr_encoder
        }
        
        # Instantiate projection head and pooler
        self.pooler = EmbedPooler(cfg.model.pooler)
        
        self.dict_projection_heads = {
            "smiles" : ProjectionHead(cfg.model.projection_head.smiles),
            "selfies" : ProjectionHead(cfg.model.projection_head.selfies),
            "graph" : ProjectionHead(cfg.model.projection_head.graph),
            "nmr" : ProjectionHead(cfg.model.projection_head.nmr)
        }
        
    def forward(self, input) -> Tensor:
        store_embeddings = {}
        # dataloader keys are other modalities
        for modality in input.keys():
            # forward through respective encoder
            store_embeddings[modality] = self.dict_encoders[modality](input[modality]).last_hidden_state
            # projection head
            store_embeddings[modality] = self.dict_projection_heads[modality](store_embeddings[modality])
        return store_embeddings