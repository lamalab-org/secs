import contextlib
from math import ceil
import chromadb
import numpy as np
import pandas as pd
from loguru import logger


def compute_retrieval_metrics_from_query(
    ids: list[str],
    retrieved_ids: list[str],
    top_k: list[int],
    output_dir: str = "./",
    modality_1: str = None,
    modality_2: str = None,
) -> dict:
    retrieval_metrics_entire_db = {k: np.zeros(len(ids)) for k in top_k}
    successful_retrievals = []
    failed_retrievals = []

    for k in top_k:
        for i, query_id in enumerate(ids):
            retrieved = retrieved_ids["ids"][i][:k]
            is_successful = query_id in retrieved
            recall_value = 1 if is_successful else 0
            retrieval_metrics_entire_db[k][i] = recall_value

            # Collect data for CSV
            record = {
                "query_id": query_id,
                "retrieved_ids": ",".join(retrieved),
                "top_k": k,
                "success": is_successful,
                "recall": recall_value,  # Add recall value
            }

            if is_successful:
                successful_retrievals.append(record)
            else:
                failed_retrievals.append(record)

    # Convert to DataFrames
    successful_df = pd.DataFrame(successful_retrievals)
    failed_df = pd.DataFrame(failed_retrievals)

    # Use modality pair in file names if provided, otherwise use generic names
    if modality_1 and modality_2:
        successful_file = f"{output_dir}/successful_retrievals_{modality_1}_{modality_2}"
        failed_file = f"{output_dir}/failed_retrievals_{modality_1}_{modality_2}"
    else:
        successful_file = f"{output_dir}/successful_retrievals.csv"
        failed_file = f"{output_dir}/failed_retrievals.csv"

    # Save to CSV files
    successful_df.to_csv(successful_file, index=False)
    failed_df.to_csv(failed_file, index=False)

    logger.info(f"Saved successful retrievals to {successful_file}")
    logger.info(f"Saved failed retrievals to {failed_file}")

    return {f"Recall@{k}": np.mean(retrieval_metrics_entire_db[k]) for k in top_k}


def batched_add(collection, embeddings, documents, ids, batch_size=5000):
    total = len(ids)
    for i in range(0, total, batch_size):
        collection.add(
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            ids=ids[i : i + batch_size],
        )


def batched_query(collection, query_embeddings, batch_size=1000, n_results=20):
    results = {"ids": []}
    for i in range(0, len(query_embeddings), batch_size):
        batch_result = collection.query(
            query_embeddings=query_embeddings[i : i + batch_size],
            n_results=n_results,
        )
        results["ids"].extend(batch_result["ids"])
    return results


def full_database_retrieval(
    embeddings: dict[str, np.ndarray],
    indices: pd.DataFrame,
    other_modalities: list[str],
    central_modality: str,
    top_k: list[int],
) -> dict[str, dict[str, float]]:
    import chromadb

    all_modalities = [central_modality, *other_modalities]
    retrieval_metrics = {}
    client = chromadb.Client()

    # Create mappings from DataFrame indices to embedding array indices for each modality
    embedding_indices = {}
    for modality in all_modalities:
        # Get indices of non-NaN rows for this modality
        valid_modality_rows = indices[modality].dropna()
        # Map DataFrame indices to embedding array positions (0-based)
        embedding_indices[modality] = {df_idx: emb_idx for emb_idx, df_idx in enumerate(valid_modality_rows.index)}

    for modality_1 in all_modalities:
        for modality_2 in all_modalities:
            if modality_2 != modality_1 and embeddings[modality_1].numel() >= embeddings[modality_2].numel():
                with contextlib.suppress(ValueError):
                    client.delete_collection(f"{modality_1}_embeds")
                collection = client.create_collection(
                    name=f"{modality_1}_embeds",
                    metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 512, "hnsw:search_ef": 1000},
                )

                # Ensure both modalities have non-NaN values for the pair
                valid_rows = indices[[modality_1, modality_2]].dropna(subset=[modality_1, modality_2])
                index_valid_rows = valid_rows.index

                # Map DataFrame indices to embedding array indices
                emb_indices_mod1 = [
                    embedding_indices[modality_1][idx] for idx in index_valid_rows if idx in embedding_indices[modality_1]
                ]
                emb_indices_mod2 = [
                    embedding_indices[modality_2][idx] for idx in index_valid_rows if idx in embedding_indices[modality_2]
                ]

                # Ensure the number of indices matches for both modalities
                if len(emb_indices_mod1) != len(emb_indices_mod2):
                    logger.warning(
                        f"Mismatch in valid indices for {modality_1} ({len(emb_indices_mod1)}) "
                        f"and {modality_2} ({len(emb_indices_mod2)}). Skipping pair."
                    )
                    continue

                # Get central modality data for valid rows
                central_modality_data = indices[central_modality][index_valid_rows].to_list()

                logger.info(f"{modality_1} {modality_2} retrieval")
                logger.info(f"Number of valid pairs: {len(central_modality_data)}")

                # Add embeddings to the collection using embedding indices
                batched_add(
                    collection=collection,
                    embeddings=embeddings[modality_1].detach().cpu().numpy()[emb_indices_mod1],
                    documents=central_modality_data,
                    ids=central_modality_data,
                )

                # Query using batched query with embedding indices
                modalities_retrieval = batched_query(
                    collection,
                    embeddings[modality_2].detach().cpu().numpy()[emb_indices_mod2],
                    n_results=max(top_k),
                )

                retrieval_metrics[f"{modality_1}_{modality_2}"] = compute_retrieval_metrics_from_query(
                    central_modality_data,
                    modalities_retrieval,
                    top_k=top_k,
                    output_dir="./",
                    modality_1=modality_1,
                    modality_2=modality_2,
                )

                logger.info(
                    f"Retrieval metrics for {modality_1} and {modality_2}: {retrieval_metrics[f'{modality_1}_{modality_2}']}"
                )

    return retrieval_metrics
