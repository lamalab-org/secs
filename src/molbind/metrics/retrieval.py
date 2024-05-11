import torch  # noqa: I002
from torch import Tensor, topk


def compute_L2_min(
    central_modality_embeddings: Tensor, other_modality_embedding: Tensor
) -> Tensor:
    return torch.norm(
        central_modality_embeddings - other_modality_embedding.unsqueeze(0), dim=1
    )


def compute_top_k_retrieval(
    embeddings_other_mod: Tensor, embeddings_central_mod: Tensor, k: int
) -> Tensor:
    in_top_k = []
    for id_embed, embedding_other_mod in enumerate(embeddings_other_mod):
        top_k = topk(
            compute_L2_min(embeddings_central_mod, embedding_other_mod),
            k,
            largest=False,
        ).indices.tolist()

        if id_embed in top_k:
            in_top_k.append(1)
        else:
            in_top_k.append(0)

    return sum(in_top_k)/len(in_top_k)
