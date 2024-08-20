import torch

from molbind.utils import select_device


def aggregate_embeddings(
    embeddings: list[dict[str, torch.Tensor]],
    modalities: list[str],
    central_modality: str,
) -> dict[str, torch.Tensor]:
    device = select_device()
    constr_dict = {modality: [] for modality in modalities}
    central_mod_embed = {}
    for embedding_dict in embeddings:
        for modality in modalities:
            if modality in embedding_dict:
                constr_dict[modality].append(embedding_dict)
    for modality, embeds in constr_dict.items():
        if modality == modalities[0]:
            central_mod_embed[central_modality] = torch.cat(
                [predict_dict[central_modality] for predict_dict in embeds], dim=0
            ).to(device)
        constr_dict[modality] = torch.cat(
            [predict_dict[modality] for predict_dict in embeds], dim=0
        ).to(device)
    constr_dict[central_modality] = central_mod_embed[central_modality]
    return constr_dict
