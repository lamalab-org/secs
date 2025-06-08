import pickle as pkl
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import polars as pl
import torch
from datasets import load_dataset
from loguru import logger
from maygen_out_to_canonical import isomer_to_canonical
from prune_sim import embedding_pruning_variable, load_models_dict, tanimoto_similarity
from tqdm.auto import tqdm

from molbind.models import MolBind
from molbind.utils.spec2struct import gen_close_molformulas_from_seed, is_neutral_no_isotopes, smiles_to_molecular_formula

ModelType = MolBind


# --- User-provided aggregate_embeddings function ---
def aggregate_embeddings_user_provided(
    embeddings: list[dict[str, torch.Tensor]],
    modalities: list[str],
) -> dict[str, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    collected_tensors_for_modality = {mod: [] for mod in modalities}

    for batch_dict in embeddings:  # Each dict is for a batch
        for mod in modalities:
            if mod in batch_dict and batch_dict[mod] is not None and batch_dict[mod].nelement() > 0:
                collected_tensors_for_modality[mod].append(batch_dict[mod].cpu())

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

        # Use the first modality in modalities_in_batch_dict as a reference for central_modality
        # This assumes 'smiles' or a similar common key is listed first if it's the intended central one.
        ref_central_modality = modalities_in_batch_dict[0] if modalities_in_batch_dict else "smiles"

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

    def load_data(self, dataset_path: str, **kwargs_raw_embeddings_paths: str | None) -> None:
        self.dataset_df = pd.read_parquet(dataset_path)
        logger.info(f"Analyzer: Loaded main dataset ({len(self.dataset_df)} entries) from {dataset_path}")

        paths_map = {s: kwargs_raw_embeddings_paths.get(f"{s}_embeddings_path") for s in self.ALL_SPECTRA_TYPES}

        current_available_modalities = []
        for spec_type in self.user_active_spectra:
            if spec_type not in self.models:
                logger.debug(f"Analyzer: Model for active spectrum {spec_type} not loaded. Skipping data for it.")
                continue

            path_str = paths_map.get(spec_type)
            if path_str and Path(path_str).exists():
                self.raw_embedding_file_paths[spec_type] = Path(path_str)
                current_available_modalities.append(spec_type)
                logger.info(f"Analyzer: Raw embedding path set for {spec_type}: {path_str}")
            else:
                logger.warning(f"Analyzer: No valid raw embedding path for {spec_type} (active spectrum with model).")

        self.available_modalities = current_available_modalities
        logger.info(f"Analyzer: Modalities with model & data path set: {self.available_modalities}")

        try:
            logger.debug("Analyzer: Attempting to load PubChem cache...")
            hf_dataset = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula", trust_remote_code=True)
            if "train" in hf_dataset:
                self.pubchem_cache = hf_dataset["train"].to_polars()
                self.pubchem_cache = self.pubchem_cache.drop_nulls(subset=["smiles", "molecular_formula"])
                is_polars_df = hasattr(self.pubchem_cache, "is_empty") and hasattr(self.pubchem_cache, "filter")
                logger.info(
                    f"Analyzer: Loaded PubChem cache. Type: {type(self.pubchem_cache)}, Is Polars DF: {is_polars_df}, Is Empty: {self.pubchem_cache.is_empty() if is_polars_df else 'N/A'}"
                )
                if is_polars_df and not self.pubchem_cache.is_empty():
                    logger.debug(f"Analyzer: PubChem cache head:\n{self.pubchem_cache.head(3)}")
                elif not is_polars_df:
                    self.pubchem_cache = None
                    logger.error("PubChem cache not a Polars DF.")
            else:
                self.pubchem_cache = None
                logger.error("PubChem 'train' split not found.")
        except Exception as e:
            logger.error(f"Analyzer: Failed to load PubChem cache: {e}", exc_info=True)
            self.pubchem_cache = None

    def _get_all_target_1D_embeddings_for_idx(self, smiles_index: int) -> dict[str, torch.Tensor]:
        target_1D_embeddings = {}
        if self.dataset_df is None or self.dataset_df.empty:
            return {}

        for spec_type in self.available_modalities:
            raw_file_path = self.raw_embedding_file_paths.get(spec_type)
            pickle_config = self.RAW_EMBEDDING_PKL_CONFIGS.get(spec_type)
            model_device = "cuda" if torch.cuda.is_available() else "cpu"

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

    def process_molecular_formula(self, smiles_index: int = 0) -> pd.DataFrame:
        if self.dataset_df is None or self.dataset_df.empty or not self.available_modalities:
            logger.error("Analyzer: Not ready (dataset None/empty or no available modalities).")
            return pd.DataFrame()

        if not (0 <= smiles_index < len(self.dataset_df)):
            logger.error(f"Analyzer: smiles_index {smiles_index} OOB for dataset (len {len(self.dataset_df)}).")
            return pd.DataFrame()

        original_smiles = self.dataset_df["smiles"].iloc[smiles_index]
        mf = smiles_to_molecular_formula(original_smiles)
        logger.info(f"Analyzer: Proc MF {mf} (idx {smiles_index}) using {self.available_modalities}")

        similar_formulas = [mf] + (gen_close_molformulas_from_seed(mf) if mf else [])

        is_valid_polars_cache = (
            self.pubchem_cache is not None
            and hasattr(self.pubchem_cache, "is_empty")
            and hasattr(self.pubchem_cache, "filter")
            and hasattr(self.pubchem_cache, "select")
            and hasattr(self.pubchem_cache, "columns")
            and bool(self.pubchem_cache.columns)
            and hasattr(self.pubchem_cache.select(self.pubchem_cache.columns[0]).to_series(), "is_in")
        )

        if not is_valid_polars_cache or self.pubchem_cache.is_empty() or not similar_formulas:
            logger.warning(
                f"Analyzer: PubChem cache not valid/empty (Valid: {is_valid_polars_cache}, Empty: {self.pubchem_cache.is_empty() if is_valid_polars_cache else 'N/A'}) or no formulas to search."
            )
            return pd.DataFrame()

        try:
            if "molecular_formula" not in self.pubchem_cache.columns:
                logger.error("Analyzer: 'molecular_formula' column missing in PubChem cache.")
                return pd.DataFrame()
            filtered_cache = self.pubchem_cache.filter(self.pubchem_cache["molecular_formula"].is_in(similar_formulas))
            # remove smiles that are contain isotopes or that are positively or negatively charged
            filtered_cache = filtered_cache.filter(pl.col("smiles").map_elements(is_neutral_no_isotopes, return_dtype=pl.Boolean))
        except Exception as e_filter:
            logger.error(f"Analyzer: Error during PubChem cache filtering: {e_filter}", exc_info=True)
            return pd.DataFrame()

        if filtered_cache is None or filtered_cache.is_empty():
            logger.warning(f"Analyzer: No PubChem hits for {mf} and variants after filtering.")
            return pd.DataFrame()

        # ... (Isomer generation, canonicalization to isomer_df as before) ...
        isomers_dir = self.cache_dir / "isomers_multiform"
        isomers_dir.mkdir(exist_ok=True, parents=True)
        isomer_file_name = f"{smiles_index}_{mf.replace(' ', '_') if mf else 'unknown_mf'}_isomers.txt"
        isomer_file = isomers_dir / isomer_file_name
        with isomer_file.open("w") as f:
            smiles_col = filtered_cache["smiles"]
            for s_str in smiles_col.to_list() if hasattr(smiles_col, "to_list") else list(smiles_col):
                f.write(s_str + "\n")
        canonical_file_name = f"{smiles_index}_{mf.replace(' ', '_') if mf else 'unknown_mf'}_canonical.csv"
        canonical_file = isomers_dir / canonical_file_name
        try:
            isomer_to_canonical(str(isomer_file), str(canonical_file))
            isomer_df = pd.read_csv(canonical_file).drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)
        except Exception as e:
            logger.error(f"Analyzer: Canonicalization error for {mf}: {e}")
            return pd.DataFrame()
        if isomer_df.empty:
            logger.warning(f"Analyzer: No canonical SMILES for {mf}.")
            return pd.DataFrame()
        num_isomers, candidate_smiles = len(isomer_df), isomer_df["canonical_smiles"].tolist()

        target_1D_embs = self._get_all_target_1D_embeddings_for_idx(smiles_index)

        isomer_df["similarity"] = [None] * num_isomers
        for st_init in self.ALL_SPECTRA_TYPES:
            isomer_df[f"{st_init}_similarity"] = [None] * num_isomers
        isomer_df["sum_of_all_individual_similarities"] = [None] * num_isomers

        if target_1D_embs:
            models_for_scoring = {s: self.models[s] for s in target_1D_embs if self.models.get(s)}
            if models_for_scoring:
                combined_sc, individual_sc_dict = embedding_pruning_variable(candidate_smiles, target_1D_embs, models_for_scoring)

                def assign(sc_tensor, num_exp, def_val=0.0):
                    if sc_tensor is not None and hasattr(sc_tensor, "numel") and sc_tensor.numel() == num_exp:
                        return sc_tensor.tolist()
                    if (
                        sc_tensor is not None and hasattr(sc_tensor, "numel") and sc_tensor.numel() == 1 and num_exp > 0
                    ):  # Should not happen with current prune_sim
                        return [sc_tensor.item()] * num_exp
                    # If sc_tensor is None (e.g. num_isomers was 0) or numel doesn't match, return default list
                    return [def_val if def_val is not None else None] * num_exp

                isomer_df["similarity"] = assign(combined_sc, num_isomers)
                current_sum = np.zeros(num_isomers, dtype=float)  # Ensure float for sum
                for st in self.ALL_SPECTRA_TYPES:
                    sc_mod = individual_sc_dict.get(st)  # This will be a tensor or None
                    assigned_scores = assign(sc_mod, num_isomers, None)
                    isomer_df[f"{st}_similarity"] = assigned_scores

                    # Summing for "sum_of_all_individual_similarities"
                    # Only sum if the modality was part of target_1D_embs (i.e., it was intended to be scored)
                    # and its scores were successfully computed (sc_mod is a tensor of correct size)
                    if st in target_1D_embs and sc_mod is not None and hasattr(sc_mod, "numel") and sc_mod.numel() == num_isomers:
                        try:
                            # assigned_scores is already a list of numbers or Nones
                            # Replace Nones with 0 for summation
                            scores_for_sum = [s if pd.notna(s) else 0.0 for s in assigned_scores]
                            current_sum += np.array(scores_for_sum, dtype=float)
                        except Exception as e_sum:
                            logger.warning(f"Could not sum scores for {st}: {e_sum}")
                isomer_df["sum_of_all_individual_similarities"] = current_sum.tolist()
            else:
                logger.warning(f"Analyzer: No models for scoring for {list(target_1D_embs.keys())}.")
        else:
            logger.error(f"Analyzer: No 1D target embeddings for idx {smiles_index}.")

        tqdm.pandas(desc=f"Analyzer: Tanimoto for {mf}")
        isomer_df["tanimoto"] = isomer_df["canonical_smiles"].progress_apply(
            lambda x: tanimoto_similarity(original_smiles, x) if pd.notna(x) and x else 0.0
        )
        # remove all different from 1
        isomer_df = isomer_df[isomer_df["tanimoto"] != 1.0].reset_index(drop=True)
        result_path = self.results_dir / f"{smiles_index}_{mf.replace(' ', '_') if mf else 'unknown_mf'}_results.csv"
        isomer_df.to_csv(result_path, index=False)
        logger.info(
            f"Analyzer: Results for {mf} (idx {smiles_index}) saved. Scored with: {list(target_1D_embs.keys()) if target_1D_embs else 'None'}"
        )
        return isomer_df

    def get_summary(self) -> dict:
        return {
            "user_active_spectra": self.user_active_spectra,
            "final_available_modalities": self.available_modalities,
            "models_loaded": list(self.models.keys()),
            "raw_emb_paths": {s: str(p) if p else "N/A" for s, p in self.raw_embedding_file_paths.items()},
            "dataset_loaded": self.dataset_df is not None,
            "dataset_size": len(self.dataset_df) if self.dataset_df is not None else 0,
            "pubchem_cache_loaded_and_valid": (
                self.pubchem_cache is not None and hasattr(self.pubchem_cache, "filter") and not self.pubchem_cache.is_empty()
            ),
        }
