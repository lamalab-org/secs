import contextlib  # noqa: I002
from typing import Dict, List  # noqa: UP035

import chromadb
import numpy as np


def full_database_retrieval(
    embeddings: Dict[str, np.ndarray],  # noqa: UP006
    ids: List[str],  # noqa: UP006
    other_modalities: List[str],  # noqa: UP006
    central_modality: str,
    top_k: List[int],  # noqa: UP006
) -> Dict[str, Dict[str, float]]:  # noqa: UP006
    client = chromadb.Client()
    with contextlib.suppress(ValueError):
        client.delete_collection(f"{central_modality}_embeds")
    collection = client.create_collection(
        name=f"{central_modality}_embeds", metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        embeddings=embeddings[central_modality].cpu().numpy(), documents=ids, ids=ids
    )
    modalities_retrieval = {}
    retrieval_metrics = {}
    for modality in other_modalities:
        modalities_retrieval[modality] = collection.query(
            query_embeddings=embeddings[modality].cpu().numpy(), n_results=max(top_k)
        )
        retrieval_metrics[modality] = compute_retrieval_metrics_from_query(
            ids, modalities_retrieval[modality], top_k=top_k
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
