import contextlib  # noqa: I002
from typing import Dict, List  # noqa: UP035

import chromadb
import numpy as np
from loguru import logger


def full_database_retrieval(
    embeddings: Dict[str, np.ndarray],  # noqa: UP006
    ids: List[str],  # noqa: UP006
    other_modalities: List[str],  # noqa: UP006
    central_modality: str,
    top_k: List[int],  # noqa: UP006
) -> Dict[str, Dict[str, float]]:  # noqa: UP006
    all_modalities = [central_modality, *other_modalities]

    retrieval_metrics = {}
    for index_mod_1, modality_1 in enumerate(all_modalities):
        client = chromadb.Client()
        with contextlib.suppress(ValueError):
            client.delete_collection(f"{modality_1}_embeds")
        collection = client.create_collection(
            name=f"{modality_1}_embeds", metadata={"hnsw:space": "cosine"}
        )
        collection.add(
            embeddings=embeddings[modality_1].cpu().numpy(),
            documents=ids,
            ids=ids,
        )
        modalities_retrieval = {}
        for index_mod_2, modality_2 in enumerate(all_modalities):
            if (
                modality_2 != modality_1
                and modality_2 != central_modality
                and index_mod_1 < index_mod_2
            ):
                modalities_retrieval[modality_2] = collection.query(
                    query_embeddings=embeddings[modality_2].cpu().numpy(),
                    n_results=max(top_k),
                )
                retrieval_metrics[f"{modality_1}_{modality_2}"] = (
                    compute_retrieval_metrics_from_query(
                        ids, modalities_retrieval[modality_2], top_k=top_k
                    )
                )
                logger.info(
                    f"Retrieval metrics for {modality_1} and {modality_2}:\
                    {retrieval_metrics[f'{modality_1}_{modality_2}']}"
                )
    return retrieval_metrics


def compute_retrieval_metrics_from_query(
    ids: List[str],  # noqa: UP006
    retrieved_ids: List[str],  # noqa: UP006
    top_k: List[int],  # noqa: UP006
) -> None:
    retrieval_metrics_entire_db = {k: np.zeros(len(ids)) for k in top_k}
    for k in top_k:
        for i, id_ in enumerate(ids):
            if id_ in retrieved_ids["ids"][i][:k]:
                retrieval_metrics_entire_db[k][i] = 1
    return {f"Recall@{k}": np.mean(retrieval_metrics_entire_db[k]) for k in top_k}
