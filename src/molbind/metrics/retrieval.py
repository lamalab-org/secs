import faiss
import numpy as np
import pandas as pd
from loguru import logger


# Helper function for L2 normalization (important for cosine similarity in Faiss)
def normalize_embeddings(embeddings_np: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    # Add a small epsilon to prevent division by zero for zero-vectors
    norms[norms == 0] = 1e-9
    return embeddings_np / norms


def compute_retrieval_metrics_from_query(
    ids: list[str],  # These are the ground truth IDs for each query
    retrieved_indices: np.ndarray,  # These are the 0-based indices from Faiss search
    indexed_ids_map: list[str],  # This list maps Faiss 0-based index to actual string ID
    top_k: list[int],
) -> dict[str, float]:
    retrieval_metrics_entire_db = {k: np.zeros(len(ids)) for k in top_k}
    num_queries = retrieved_indices.shape[0]

    for k_val in top_k:
        for i in range(num_queries):
            # The ground truth ID for the i-th query
            ground_truth_id_for_query = ids[i]
            # Get the string IDs of the top-k retrieved items for this query
            # retrieved_indices[i, :k_val] are the Faiss indices
            current_retrieved_string_ids = [
                indexed_ids_map[idx] for idx in retrieved_indices[i, :k_val] if idx != -1
            ]  # -1 if less than k results

            if ground_truth_id_for_query in current_retrieved_string_ids:
                retrieval_metrics_entire_db[k_val][i] = 1

    return {f"Recall@{k_val}": np.mean(retrieval_metrics_entire_db[k_val]) for k_val in top_k}


def full_database_retrieval(
    embeddings: dict[str, np.ndarray],  # Assuming these are PyTorch tensors
    indices: pd.DataFrame,  # DataFrame mapping modalities to item IDs
    other_modalities: list[str],
    central_modality: str,
    top_k: list[int],
) -> dict[str, dict[str, float]]:
    indices = indices.reset_index(drop=True)  # Ensure indices are reset for consistency
    all_modalities = [central_modality, *other_modalities]
    max_k = max(top_k)
    retrieval_metrics = {}
    for _, modality_1 in enumerate(all_modalities):  # Modality to build the Faiss index from
        for _, modality_2 in enumerate(all_modalities):  # Modality to query with
            if modality_2 == modality_1:
                continue

            logger.info(f"Processing: Index with {modality_1}, Query with {modality_2}")

            # --- Data Preparation ---
            # Find common items that have valid IDs in modality_1, modality_2, and central_modality
            # These columns in 'indices' DataFrame are assumed to hold the *identifiers* for items
            # in each modality, or be NaN if an item doesn't exist for that modality.
            # The actual embeddings are in the `embeddings` dict, indexed by row number.

            # Get original DataFrame indices for rows that have non-NaN values for all relevant ID columns
            # This ensures we are working with a consistent set of items across modalities being compared.
            common_item_original_indices = indices[[modality_1, modality_2, central_modality]].dropna().index

            # Get the actual embeddings for these common items
            # These are the vectors that will go into the Faiss index
            db_vectors_np = embeddings[modality_1].detach().cpu().numpy()[common_item_original_indices]
            # These are the vectors that will be used for querying
            query_vectors_np = embeddings[modality_2].detach().cpu().numpy()[common_item_original_indices]

            # These are the ground truth IDs from the central modality for the common items.
            # This list serves two purposes:
            # 1. It's the list of IDs that Faiss will be effectively storing (by position).
            # 2. It's the list of ground truth IDs for the queries (since we query for the corresponding item).
            #    e.g. query_vectors_np[i] should ideally retrieve db_vectors_np[i],
            #    and the ID for this pair is ground_truth_ids_for_common_items[i].
            ground_truth_ids_for_common_items = indices.loc[common_item_original_indices, central_modality].tolist()

            if not (len(db_vectors_np) == len(query_vectors_np) == len(ground_truth_ids_for_common_items)):
                logger.error("Mismatch in lengths of prepared data. This should not happen. Skipping.")
                logger.error(
                    f"DB vecs: {len(db_vectors_np)}, Query vecs: {len(query_vectors_np)}, GT IDs: {len(ground_truth_ids_for_common_items)}"
                )
                continue

            if len(db_vectors_np) == 0:
                logger.info(f"No data to index for {modality_1} after filtering. Skipping.")
                continue

            logger.info(
                f"Number of items for {modality_1} (DB) and {modality_2} (Query) pair: {len(ground_truth_ids_for_common_items)}"
            )

            # Normalize embeddings for cosine similarity
            db_vectors_normalized = normalize_embeddings(db_vectors_np.astype(np.float32))
            query_vectors_normalized = normalize_embeddings(query_vectors_np.astype(np.float32))

            dimension = db_vectors_normalized.shape[1]

            # --- Faiss Index Setup ---
            # Using HNSW for approximate nearest neighbor search, with Inner Product (cosine)
            # M is the number of connections per layer in HNSW
            # efConstruction is the depth of exploration during index construction
            # efSearch is the depth of exploration during search
            # M = 32  # A common value for HNSW, can be tuned
            index = faiss.IndexFlatIP(dimension)
            # index.hnsw.efConstruction = 2000
            # index.hnsw.efSearch = 128

            # For exact search (slower, especially for large datasets):
            # index = faiss.IndexFlatIP(dimension)

            if not index.is_trained:  # HNSWFlat doesn't require separate training if base is Flat
                index.train(db_vectors_normalized)  # Not strictly necessary for HNSWFlat but good practice

            index.add(db_vectors_normalized)
            logger.info(f"Faiss index built for {modality_1} with {index.ntotal} vectors.")

            # --- Querying ---
            # D are distances (inner products), I are the 0-based indices of retrieved vectors
            # For METRIC_INNER_PRODUCT with normalized vectors, higher D is better (closer to 1 for identical)
            D, I = index.search(query_vectors_normalized, max_k)

            # --- Metric Calculation ---
            # The `compute_retrieval_metrics_from_query` function needs:
            # 1. `ids`: The ground truth ID for each query vector. In this setup,
            #    query_vectors_normalized[j] corresponds to ground_truth_ids_for_common_items[j].
            # 2. `retrieved_indices`: The `I` from faiss.search.
            # 3. `indexed_ids_map`: A list that maps the 0-based index from Faiss back to the
            #    actual string ID. This is `ground_truth_ids_for_common_items` because the i-th vector
            #    added to the index corresponds to `ground_truth_ids_for_common_items[i]`.

            current_metrics = compute_retrieval_metrics_from_query(
                ids=ground_truth_ids_for_common_items,  # Ground truth ID for each query
                retrieved_indices=I,  # Faiss indices of retrieved items
                indexed_ids_map=ground_truth_ids_for_common_items,  # Map Faiss index to string ID
                top_k=top_k,
            )
            metric_key = f"{modality_1}_DB__{modality_2}_Query"
            retrieval_metrics[metric_key] = current_metrics
            logger.info(f"Retrieval metrics for {metric_key}: {retrieval_metrics[metric_key]}")
            # Faiss index is in memory, will be garbage collected or overwritten in next loop
        del index

    return retrieval_metrics
