from typing import Tuple
from transformers import AutoModelForCausalLM
import torch.nn as nn
from torch import Tensor


def xavier_init(model: nn.Module):
    for param in model.parameters():
        if len(param.shape) > 1:
            nn.init.xavier_uniform_(param)
    return model


class BaseModalityEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_encoder: bool = False,
        pretrained: bool = True,
        **kwargs,
    ):
        super(BaseModalityEncoder, self).__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.pretrained = pretrained
        self.encoder = None
        self._initialize_encoder()

    def _initialize_encoder(self):
        if self.pretrained:
            self.encoder = AutoModelForCausalLM.from_pretrained(self.model_name)
            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.encoder = xavier_init(self.encoder)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        token_ids, attention_mask = x
        output = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden_state = output.hidden_states[-1]

        return self._non_pad_token_embed_averaging(last_hidden_state, attention_mask)

    def _non_pad_token_embed_averaging(
        self, last_hidden_state: Tensor, attention_mask: Tensor
    ) -> Tensor:
        attention_mask = attention_mask.float().unsqueeze(-1)
        sum_ = (last_hidden_state * attention_mask).sum(dim=1)
        norm = attention_mask.squeeze(-1).sum(dim=1).unsqueeze(1)
        return sum_ / norm
