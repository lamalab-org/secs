"""
Sources: https://towardsdatascience.com/adding-custom-layers-on-top-of-a-hugging-face-model-f1ccdfc257bd
"""

from types import SimpleNamespace
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import os


ModalityType = SimpleNamespace(
    SMILES="smiles",
    SELFIES="selfies",
    TEXT="text",
    NMR="nmr",
    STRUCTURES_3D="structures_3d",
    IMAGES_2D="images_2d",
    IR="ir"
)

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MolBind(nn.Module):
    def __init__(self, model_opt_params : Dict):
        super().__init__()
    
        # SMILES is the central modality
        self.modality_pair = (ModalityType.SMILES, self.config.modality_2)
    
        # SMILES model
        self.smiles_model = AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").requires_grad_(False)
        self.tokenizer_smiles = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.dropout_smiles = nn.Dropout(0.1)
        self.fc_smiles = nn.Sequential(
            nn.Linear(768, 256), 
            nn.LeakyReLU(),
            nn.Linear(256, 1024)
        )
        
        # SELFIES model
        self.selfies_model = AutoModelForCausalLM.from_pretrained("HUBioDataLab/SELFormer").requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained("HUBioDataLab/SELFormer")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1024)
        )
        # NMR model
        
        # IR model
        
        # Text model (IUPAC names, etc.)
        
        # Image model (2D structures)
        
        # 3D model (3D structures)
        
    def forward(self, input_ids_modality_1, attention_mask_modality_1, input_ids_modality_2, attention_mask_modality_2):
        
        output_chemberta = self.smiles_model(input_ids=input_ids_modality_1, 
                                            attention_mask=attention_mask_modality_1,
                                            output_hidden_states=True)
        hidden_states_chemberta = output_chemberta.hidden_states[1][:, -1]
        x_chemberta = self.fc_smiles(hidden_states_chemberta)
        
        if self.modality_pair[1] == "selfies":
            outputs_selformer = self.selfies_model(
                input_ids=input_ids_modality_2, 
                attention_mask=attention_mask_modality_2,  
                output_hidden_states=True
                )
            # RETURNED HIDDEN STATES
            hidden_states_selformer = outputs_selformer.hidden_states[1][:, -1]
            x = self.fc(hidden_states_selformer)
        return x, x_chemberta
    
    def load_from_checkpoint(self, path: str):
        return torch.load(path)