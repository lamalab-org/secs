import pickle as pkl
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import polars as pl
import torch
from loguru import logger
from prune import embedding_pruning_variable, load_models_dict

from molbind.models import MolBind
from molbind.utils.spec2struct import gen_close_molformulas_from_seed, is_neutral_no_isotopes

ModelType = MolBind


def aggregate_embeddings_user_provided(
    embeddings: list[dict[str, torch.Tensor]],
    modalities: list[str],
) -> dict[str, torch.Tensor]:
    device = "cpu"
    collected_tensors_for_modality = {mod: [] for mod in modalities}

    for batch_dict in embeddings:  # Each dict is for a batch
        for mod in modalities:
            if mod in batch_dict and batch_dict[mod] is not None and batch_dict[mod].nelement() > 0:
                collected_tensors_for_modality[mod].append(batch_dict[mod])

    concatenated_embeddings = {}
    for mod, tensor_list in collected_tensors_for_modality.items():
        if tensor_list:
            try:
                concatenated_embeddings[mod] = torch.cat(tensor_list, dim=0).to(device)
                logger.debug(f"Aggregated '{mod}', final shape: {concatenated_embeddings[mod].shape}")
            except Exception as e_cat:
                logger.error(f"aggregate_embeddings: Error concatenating for '{mod}': {e_cat}")
                concatenated_embeddings[mod] = torch.empty(0, device=device)
        else:
            logger.warning(f"aggregate_embeddings: No tensors collected for '{mod}'.")
            concatenated_embeddings[mod] = torch.empty(0, device=device)
    return concatenated_embeddings


def get_1d_target_embedding_from_raw_batches_pkl(
    raw_embedding_file_path: str,
    target_idx: int,
    pickle_content_config: dict,  # Must contain 'modalities_in_batch_dict' and 'primary_spectral_key'
    expected_total_molecules_after_aggregation: int,
    device: str = "cpu",
) -> torch.Tensor | None:
    modalities_in_batch_dict = pickle_content_config.get("modalities_in_batch_dict")
    primary_spec_key = pickle_content_config.get("primary_spectral_key")

    if not modalities_in_batch_dict or not primary_spec_key:
        logger.error(f"Config error for {raw_embedding_file_path}.")
        return None

    try:
        with open(raw_embedding_file_path, "rb") as f:
            list_of_batch_dicts = pkl.load(f)
        if not isinstance(list_of_batch_dicts, list) or not list_of_batch_dicts:
            logger.error(f"{raw_embedding_file_path} not a valid list of batch dicts.")
            return None

        aggregated_data = aggregate_embeddings_user_provided(embeddings=list_of_batch_dicts, modalities=modalities_in_batch_dict)

        all_spectral_embs = aggregated_data.get(primary_spec_key)
        if all_spectral_embs is None or all_spectral_embs.nelement() == 0:
            logger.warning(f"No aggregated '{primary_spec_key}' from {raw_embedding_file_path}.")
            return None

        if all_spectral_embs.shape[0] != expected_total_molecules_after_aggregation:
            logger.error(
                f"Data Mismatch: Aggregated '{primary_spec_key}' from {raw_embedding_file_path} has {all_spectral_embs.shape[0]} entries, expected {expected_total_molecules_after_aggregation}."
            )
            return None

        if not (0 <= target_idx < all_spectral_embs.shape[0]):
            logger.error(f"Target idx {target_idx} OOB for aggregated data (len {all_spectral_embs.shape[0]}).")
            return None

        target_mol_emb = all_spectral_embs[target_idx].to(device)

        final_1D_tensor = target_mol_emb
        # Step 1: Squeeze if there's a leading batch-like dimension of 1 from the per-molecule slice
        # This handles cases where aggregate_embeddings might produce (N, 1, L, D) and indexing gives (1, L, D)
        if final_1D_tensor.ndim > 1 and final_1D_tensor.shape[0] == 1:
            final_1D_tensor = final_1D_tensor.squeeze(0)
        # Step 2: If now 2D (L,D) (sequence), aggregate by mean pooling
        if final_1D_tensor.ndim == 2:
            final_1D_tensor = torch.mean(final_1D_tensor, dim=0)

        if final_1D_tensor.ndim == 1:
            return final_1D_tensor
        else:
            logger.error(
                f"Could not reduce '{primary_spec_key}' (idx {target_idx}) to 1D. Initial shape: {target_mol_emb.shape}, Final shape: {final_1D_tensor.shape}"
            )
            return None

    except FileNotFoundError:
        logger.warning(f"Not found: {raw_embedding_file_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing {raw_embedding_file_path} for idx {target_idx}: {e}", exc_info=True)
        return None


class SimpleMoleculeAnalyzer:
    ALL_SPECTRA_TYPES: ClassVar[list[str]] = ["ir", "cnmr", "hnmr", "hsqc"]
    RAW_EMBEDDING_PKL_CONFIGS: ClassVar[dict[str, dict]] = {
        "ir": {"modalities_in_batch_dict": ["smiles", "ir"], "primary_spectral_key": "ir"},
        "cnmr": {"modalities_in_batch_dict": ["smiles", "c_nmr"], "primary_spectral_key": "c_nmr"},
        "hnmr": {"modalities_in_batch_dict": ["smiles", "h_nmr"], "primary_spectral_key": "h_nmr"},
        "hsqc": {"modalities_in_batch_dict": ["smiles", "hsqc"], "primary_spectral_key": "hsqc"},
    }

    def __init__(self, models_config_path: str, results_dir: str, cache_dir: str | None, active_spectra: list[str] | None):
        self.models_config_path = Path(models_config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.user_active_spectra = [s for s in (active_spectra or self.ALL_SPECTRA_TYPES) if s in self.ALL_SPECTRA_TYPES]

        self.models: dict[str, ModelType | None] = {}
        self.raw_embedding_file_paths: dict[str, Path | None] = {}
        self.dataset_df: pd.DataFrame | None = None
        self.pubchem_cache: Any | None = None  # Polars DataFrame
        self.available_modalities: list[str] = []

    def load_models(self, **kwargs_experiments: str | None) -> None:
        exp_dict = {s: kwargs_experiments.get(f"{s}_experiment") for s in self.ALL_SPECTRA_TYPES}
        loaded_models = load_models_dict(str(self.models_config_path), exp_dict)
        self.models = {s: model for s, model in loaded_models.items() if model}
        logger.info(f"Analyzer: Models loaded for: {list(self.models.keys())}")

    def load_pubchem(self) -> None:
        hf_dataset = pl.read_parquet("cache_pubchem.parquet")
        self.pubchem_cache = hf_dataset.drop_nulls(subset=["smiles", "molecular_formula"])

    def _get_all_target_1D_embeddings_for_idx(self, smiles_index: int) -> dict[str, torch.Tensor]:
        target_1D_embeddings = {}
        if self.dataset_df is None or self.dataset_df.empty:
            return {}

        for spec_type in self.available_modalities:
            raw_file_path = self.raw_embedding_file_paths.get(spec_type)
            pickle_config = self.RAW_EMBEDDING_PKL_CONFIGS.get(spec_type)
            model_device = "cpu"

            if raw_file_path and pickle_config:
                emb_tensor = get_1d_target_embedding_from_raw_batches_pkl(
                    raw_embedding_file_path=str(raw_file_path),
                    target_idx=smiles_index,
                    pickle_content_config=pickle_config,
                    expected_total_molecules_after_aggregation=len(self.dataset_df),
                    device=model_device,
                )
                if emb_tensor is not None:
                    target_1D_embeddings[spec_type] = emb_tensor
        return target_1D_embeddings

    def _get_all_target_1D_embeddings_for_smiles(self, smiles: str) -> dict[str, torch.Tensor]:
        """Generates 1D embeddings for a given SMILES string on-the-fly."""
        target_1D_embeddings = {}
        device = "cpu"

        for spec_type in self.available_modalities:
            model = self.models.get(spec_type)
            if not model:
                logger.debug(f"No model available for {spec_type}, skipping embedding generation.")
                continue

            try:
                input_batch = {"smiles": [smiles]}
                primary_spec_key = self.RAW_EMBEDDING_PKL_CONFIGS[spec_type]["primary_spectral_key"]
                modalities_to_encode = ["smiles", primary_spec_key]

                with torch.no_grad():
                    model.to(device)
                    model.eval()
                    encoded_output = model.encode(input_batch, modalities=modalities_to_encode)

                emb_tensor = encoded_output.get(primary_spec_key)

                if emb_tensor is None or emb_tensor.nelement() == 0:
                    logger.warning(f"On-the-fly embedding for {spec_type} was empty for SMILES: {smiles}")
                    continue

                final_1D_tensor = emb_tensor
                if final_1D_tensor.ndim > 1 and final_1D_tensor.shape[0] == 1:
                    final_1D_tensor = final_1D_tensor.squeeze(0)
                if final_1D_tensor.ndim == 2:  # Sequence, e.g., (L, D)
                    final_1D_tensor = torch.mean(final_1D_tensor, dim=0)

                if final_1D_tensor.ndim == 1:
                    target_1D_embeddings[spec_type] = final_1D_tensor.to(device)
                else:
                    logger.error(
                        f"Could not reduce on-the-fly embedding for {spec_type} to 1D. Final shape: {final_1D_tensor.shape}"
                    )

            except Exception as e:
                logger.error(f"Failed to generate on-the-fly embedding for {spec_type} with SMILES {smiles}: {e}", exc_info=True)

        return target_1D_embeddings

    def process_from_molecular_formula(
        self,
        mf_str: str,
        target_embeddings: dict[str, torch.Tensor],
    ) -> pd.DataFrame:
        """
        Finds and ranks isomers for a molecular formula given user-provided spectral embeddings.

        This function performs structure elucidation by searching a database for all isomers
        of a given formula and scoring them against the provided embeddings. All operations
        are performed in-memory.

        Args:
            mf_dict: The molecular formula, e.g., {"C": 10, "H": 12, "N": 2}.
            target_embeddings: A dict of 1D torch.Tensor embeddings, e.g.,
                               {'hnmr': tensor_h, 'cnmr': tensor_c}.
            run_name: An optional unique name for this run, used for in-memory caching.

        Returns:
            A pandas DataFrame of candidate isomers ranked by similarity, or an empty
            DataFrame on failure.
        """

        # --- 1. Input Validation and Pre-computation (Guard Clauses) ---

        # mf_str = "".join(f"{k}{v}" for k, v in sorted(mf_dict.items()) if v > 1)

        models_for_scoring = {st: self.models[st] for st in target_embeddings if st in self.models}
        close_mol_formulas = gen_close_molformulas_from_seed(mf_str)
        filtered_cache = self.pubchem_cache.filter(pl.col("molecular_formula").is_in(close_mol_formulas))
        filtered_cache = filtered_cache.filter(pl.col("smiles").map_elements(is_neutral_no_isotopes, return_dtype=pl.Boolean))
        # Directly get SMILES list from the Polars DataFrame
        isomer_df = filtered_cache.to_pandas()

        # --- 4. Scoring ---
        num_isomers, candidate_smiles = len(isomer_df), isomer_df["smiles"].tolist()
        logger.info(f"Scoring {num_isomers} candidate isomers for {mf_str}.")

        combined_sc, individual_sc_dict = embedding_pruning_variable(candidate_smiles, target_embeddings, models_for_scoring)

        def assign(sc_tensor, num_exp):
            return sc_tensor.tolist() if sc_tensor is not None and sc_tensor.numel() == num_exp else [np.nan] * num_exp

        isomer_df["similarity"] = assign(combined_sc, num_isomers)
        current_sum = np.zeros(num_isomers, dtype=float)

        for st in self.ALL_SPECTRA_TYPES:
            scores = assign(individual_sc_dict.get(st), num_isomers)
            isomer_df[f"{st}_similarity"] = scores
            if st in models_for_scoring:
                current_sum += np.nan_to_num(np.array(scores, dtype=float))

        isomer_df["sum_of_all_individual_similarities"] = current_sum
        return isomer_df.sort_values(by="similarity", ascending=False).reset_index(drop=True)
