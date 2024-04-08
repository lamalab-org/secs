from typing import Dict
import torch
from transformers import AutoModelForCausalLM
from molbind.models.components.base_encoder import BaseModalityEncoder
from molbind.utils.utils import reinitialize_weights


class SmilesEncoder(BaseModalityEncoder):
    def __init__(self,
                freeze_encoder : bool = False,
                pretrained=True,
                **kwargs) -> None:
        super().__init__(freeze_encoder, pretrained, **kwargs)
        self.encoder = AutoModelForCausalLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not pretrained:
            self.encoder = reinitialize_weights(self.encoder)

    def forward(self, x):
        token_ids, attention_mask = x
        return self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
            ).hidden_states[-1].sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(1)


class SelfiesEncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained=True, **kwargs) -> None:
        super().__init__(freeze_encoder, pretrained, **kwargs)
        self.encoder = AutoModelForCausalLM.from_pretrained("HUBioDataLab/SELFormer")
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not pretrained:
            self.encoder = reinitialize_weights(self.encoder)

    def forward(self, x : Dict[str, torch.Tensor]) -> torch.Tensor:
        token_ids, attention_mask = x

        return self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
            ).hidden_states[-1].sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(1)


class IUPACNameEncoder(BaseModalityEncoder):
    pass


class IREncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained: bool = False, **kwargs) -> None:
        super().__init__(freeze_encoder, pretrained, **kwargs)
        self.encoder = None
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if pretrained:
            self.encoder = reinitialize_weights(self.encoder)

    def forward(self, x):
        token_ids, attention_mask = x
        return self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
            ).hidden_states[-1].sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(1)


class GraphEncoder(BaseModalityEncoder):
    pass


class NMREncoder(BaseModalityEncoder):
    def __init__(self, freeze_encoder: bool = False, pretrained=True, **kwargs):
        super().__init__(freeze_encoder, pretrained, **kwargs)
        self.encoder = None
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not pretrained:
            self.encoder = reinitialize_weights(self.encoder)
    
    def forward(self, x):
        token_ids, attention_mask = x
        return self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
            ).hidden_states[-1].sum(dim=1)/attention_mask.sum(dim=1).unsqueeze(1)