import contextlib

import chromadb
import numpy as np
import pandas as pd
from loguru import logger


def compute_retrieval_metrics_from_query(
    ids: list[str],
    retrieved_ids: list[str],
    top_k: list[int],
) -> None:
    retrieval_metrics_entire_db = {k: np.zeros(len(ids)) for k in top_k}
    for k in top_k:
        for i, id_ in enumerate(ids):
            if id_ in retrieved_ids["ids"][i][:k]:
                retrieval_metrics_entire_db[k][i] = 1
    return {f"Recall@{k}": np.mean(retrieval_metrics_entire_db[k]) for k in top_k}


def full_database_retrieval(
    embeddings: dict[str, np.ndarray],
    indices: pd.DataFrame,
    other_modalities: list[str],
    central_modality: str,
    top_k: list[int],
) -> dict[str, dict[str, float]]:
    all_modalities = [central_modality, *other_modalities]

    retrieval_metrics = {}
    client = chromadb.Client()

    for _, modality_1 in enumerate(all_modalities):
        modalities_retrieval = {}
        for _, modality_2 in enumerate(all_modalities):
            if modality_2 != modality_1 and embeddings[modality_1].size() >= embeddings[modality_2].size():
                with contextlib.suppress(ValueError):
                    client.delete_collection(f"{modality_1}_embeds")
                collection = client.create_collection(name=f"{modality_1}_embeds", metadata={"hnsw:space": "cosine"})

                embed_mod1 = indices[[modality_1, modality_2]].dropna(subset=modality_1).reset_index().dropna(subset=modality_2)
                embed_mod2 = indices[[modality_1, modality_2]].dropna(subset=modality_2).reset_index().dropna(subset=modality_1)
                index_embed_mod1 = embed_mod1.index
                index_embed_mod2 = embed_mod2.index
                index_central_mod = indices[[modality_1, modality_2]].dropna().index.to_list()
                central_modality_data = indices[central_modality][index_central_mod].to_list()
                logger.info(f"{modality_1} {modality_2} retrieval")
                logger.info(f"{len(central_modality_data)}")
                collection.add(
                    embeddings=embeddings[modality_1].detach().cpu().numpy()[index_embed_mod1],
                    documents=central_modality_data,
                    ids=central_modality_data,
                )
                modalities_retrieval[modality_2] = collection.query(
                    query_embeddings=embeddings[modality_2].detach().cpu().numpy()[index_embed_mod2],
                    n_results=max(top_k),
                )
                retrieval_metrics[f"{modality_1}_{modality_2}"] = compute_retrieval_metrics_from_query(
                    central_modality_data,
                    modalities_retrieval[modality_2],
                    top_k=top_k,
                )
                logger.info(
                    f"Retrieval metrics for {modality_1} and {modality_2}:\
                    {retrieval_metrics[f'{modality_1}_{modality_2}']}"
                )
    return retrieval_metrics
