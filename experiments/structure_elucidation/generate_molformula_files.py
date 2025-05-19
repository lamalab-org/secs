from functools import partial
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from datasets import load_dataset
from loguru import logger
from maygen_out_to_canonical import isomer_to_canonical
from prune_sim import embedding_pruning, load_models, tanimoto_similarity
from tqdm.auto import tqdm

from molbind.data.analysis.utils import aggregate_embeddings
from molbind.utils.spec2struct import gen_close_molformulas_from_seed, smiles_to_molecular_formula

ModelType = Any


class SimpleMoleculeAnalyzer:
    """
    A user-friendly class for analyzing molecular structures using spectroscopic data.

    This class simplifies the process of analyzing molecular structures by providing
    a clean interface to the underlying functionality. It allows customization of
    which spectra (from IR, CNMR, HNMR) are actively used in similarity calculations.
    """

    # Define known spectra types and their corresponding modality keys used in embeddings
    ALL_SPECTRA_TYPES: ClassVar[list[str]] = ["ir", "cnmr", "hnmr"]
    MODALITY_MAP: ClassVar[dict[str, str]] = {
        "ir": "ir",
        "cnmr": "c_nmr",
        "hnmr": "h_nmr_cnn",
    }

    def __init__(
        self,
        models_config_path: str = "../../configs",
        results_dir: str = "analysis_results",
        cache_dir: str | None = None,
        active_spectra: list[str] | None = None,
    ):
        """
        Initialize the SimpleMoleculeAnalyzer.

        Args:
            models_config_path: Path to the models configuration directory.
            data_dir: Directory containing datasets and embeddings.
            results_dir: Directory where results will be saved.
            cache_dir: Directory for cached data (will use default if None).
            active_spectra: list of spectra types to use (e.g., ["ir", "cnmr"]).
                            Defaults to all available spectra (["ir", "cnmr", "hnmr"]).
                            These spectra must be among "ir", "cnmr", "hnmr".
        """
        self.models_config_path = Path(models_config_path)

        self.results_dir = Path(results_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / "cache"

        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        if active_spectra is None:
            self.active_spectra = self.ALL_SPECTRA_TYPES[:]
        else:
            self.active_spectra = []
            for spec_type in active_spectra:
                if spec_type not in self.ALL_SPECTRA_TYPES:
                    logger.warning(
                        f"Unknown spectrum type '{spec_type}' in active_spectra. Ignoring. "
                        f"Allowed types are: {self.ALL_SPECTRA_TYPES}"
                    )
                else:
                    self.active_spectra.append(spec_type)

        logger.info(f"Analyzer initialized. Active spectra: {self.active_spectra}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")

        # Initialize models and embeddings storage
        self.models: dict[str, ModelType | None] = dict.fromkeys(self.ALL_SPECTRA_TYPES)
        self.embeddings_data: dict[str, pd.DataFrame | None] = dict.fromkeys(self.ALL_SPECTRA_TYPES)
        self.dataset: pd.DataFrame | None = None
        self.pubchem_cache: Any | None = None  # Type depends on to_polars() output

        # Paths for loaded data (set in load_data)
        self.dataset_path: Path | None = None
        self.embedding_paths: dict[str, Path | None] = dict.fromkeys(self.ALL_SPECTRA_TYPES)

    def load_models(
        self,
        ir_experiment: str = "metrics/ir_simulated_large_dataset_testset",
        cnmr_experiment: str = "metrics/cnmr_simulated_large_dataset_testset",
        hnmr_experiment: str = "metrics/hnmr_cnn_detailed_large_dataset_testset",
    ) -> None:
        """
        Load the spectroscopic models for IR, CNMR, and HNMR.

        All three models are loaded as they are required by the underlying
        `embedding_pruning` function. Inactive spectra (not in `self.active_spectra`)
        will have their weights set to zero during processing.

        Args:
            ir_experiment: Name of the IR experiment.
            cnmr_experiment: Name of the CNMR experiment.
            hnmr_experiment: Name of the HNMR experiment.
        """

        loaded_ir_model, loaded_cnmr_model, loaded_hnmr_model = load_models(
            self.models_config_path,
            ir_experiment=ir_experiment,
            cnmr_experiment=cnmr_experiment,
            hnmr_experiment=hnmr_experiment,
        )
        self.models["ir"] = loaded_ir_model
        self.models["cnmr"] = loaded_cnmr_model
        self.models["hnmr"] = loaded_hnmr_model

    def load_data(self, dataset_path: str, ir_embeddings_path: str, cnmr_embeddings_path: str, hnmr_embeddings_path: str) -> None:
        """
        Load datasets and embeddings for IR, CNMR, and HNMR.

        Args:
            dataset_path: Path to the dataset pickle file.
            ir_embeddings_path: Path to the IR embeddings pickle file.
            cnmr_embeddings_path: Path to the CNMR embeddings pickle file.
            hnmr_embeddings_path: Path to the HNMR embeddings pickle file.
        """
        self.dataset_path = Path(dataset_path)

        paths_map = {"ir": Path(ir_embeddings_path), "cnmr": Path(cnmr_embeddings_path), "hnmr": Path(hnmr_embeddings_path)}
        # check format of dataset_path
        suffix = self.dataset_path.suffix
        self.dataset = pd.read_pickle(self.dataset_path) if suffix == ".pkl" else pd.read_parquet(self.dataset_path)
        # rename h_nmr to h_nmr_cnn
        if "h_nmr" in self.dataset.columns:
            self.dataset = self.dataset.rename(columns={"h_nmr": "h_nmr_cnn"})

        for spec_type in self.ALL_SPECTRA_TYPES:
            path = paths_map[spec_type]
            self.embedding_paths[spec_type] = path
            self.embeddings_data[spec_type] = pd.read_pickle(path)

        self.pubchem_cache = load_dataset("jablonkagroup/pubchem-smiles-molecular-formula")["train"].to_polars()
        self.pubchem_cache = self.pubchem_cache.drop_nulls()

    def process_molecular_formula(self, smiles_index: int = 0) -> pd.DataFrame:
        """
        Process a molecular formula to find similar structures.

        Args:
            smiles_index: Index of the reference SMILES in the dataset.

        Returns:
            DataFrame with similarity results.
        """

        molecular_formula = smiles_to_molecular_formula(self.dataset["smiles"].iloc[smiles_index])

        logger.info(f"Processing molecular formula: {molecular_formula}")
        similar_formulas = [molecular_formula, *gen_close_molformulas_from_seed(molecular_formula)]
        filtered_cache = self._filter_by_molecular_formulas(self.pubchem_cache, similar_formulas)

        isomers_dir = self.cache_dir / "isomers_multiform"
        isomers_dir.mkdir(exist_ok=True, parents=True)
        isomer_file = isomers_dir / f"{molecular_formula}.txt"
        with isomer_file.open("w") as f:
            for smiles in filtered_cache["smiles"]:
                f.write(smiles + "\n")

        canonical_file = isomers_dir / f"{smiles_index}_{molecular_formula}.csv"
        isomer_to_canonical(str(isomer_file), str(canonical_file))
        isomer_df = pd.read_csv(canonical_file)
        isomer_df = isomer_df.drop_duplicates(subset=["canonical_smiles"])

        # --- 4. Aggregate embeddings and get reference embeddings ---
        aggregated_embeddings_store: dict[str, Any] = {}

        for spec_type in self.ALL_SPECTRA_TYPES:
            modalities = ["smiles", self.MODALITY_MAP[spec_type]]
            agg_emb = aggregate_embeddings(
                embeddings=self.embeddings_data[spec_type],
                modalities=modalities,
                central_modality="smiles",
            )
            aggregated_embeddings_store[spec_type] = agg_emb

        # perform embedding pruning
        results_tuple = embedding_pruning(
            spectra_ir_embedding=aggregated_embeddings_store["ir"]["ir"][smiles_index],
            spectra_cnmr_embedding=aggregated_embeddings_store["cnmr"]["c_nmr"][smiles_index],
            spectra_hnmr_embedding=aggregated_embeddings_store["hnmr"]["h_nmr_cnn"][smiles_index],
            ir_model=self.models["ir"],
            cnmr_model=self.models["cnmr"],
            hnmr_model=self.models["hnmr"],
            smiles=isomer_df["canonical_smiles"].to_list(),
        )
        # calculate similarities
        (
            cosine_similarities,
            ir_similarities,
            cnmr_similarities,
            hnmr_similarities,
        ) = results_tuple

        def to_list_if_tensor(x):
            return x.tolist() if hasattr(x, "tolist") else x

        isomer_df["similarity"] = to_list_if_tensor(cosine_similarities)
        isomer_df["ir_similarity"] = to_list_if_tensor(ir_similarities)
        isomer_df["cnmr_similarity"] = to_list_if_tensor(cnmr_similarities)
        isomer_df["hnmr_similarity"] = to_list_if_tensor(hnmr_similarities)

        isomer_df["sum_of_all_individual_similarities"] = (
            isomer_df["ir_similarity"].fillna(0) + isomer_df["cnmr_similarity"].fillna(0) + isomer_df["hnmr_similarity"].fillna(0)
        )
        original_smiles = self.dataset.smiles.to_list()[smiles_index]
        tanimoto_func = partial(tanimoto_similarity, original_smiles)
        tqdm.pandas(desc="Calculating Tanimoto similarity")
        isomer_df["tanimoto"] = isomer_df["canonical_smiles"].progress_apply(lambda x: tanimoto_func(x) if pd.notna(x) else None)

        # --- 7. Save results ---
        result_path = self.results_dir / f"{smiles_index}_{molecular_formula}.csv"
        isomer_df.to_csv(result_path, index=False)
        logger.info(f"Results for {molecular_formula} saved to {result_path}")

        return isomer_df

    def analyze_multiple_formulas(
        self,
        molecular_formulas: list[str],
        start_index: int = 0,
    ) -> dict[str, pd.DataFrame]:
        """
        Analyze multiple molecular formulas.

        Args:
            molecular_formulas: list of molecular formulas to analyze.
            start_index: Starting index for SMILES references in the dataset.

        Returns:
            Dictionary mapping each formula to its result DataFrame.
        """
        results_dict = {}
        for i, formula in enumerate(tqdm(molecular_formulas, desc="Processing formulas")):
            logger.info(f"Processing formula {i + 1}/{len(molecular_formulas)}: {formula}")
            current_smiles_index = start_index + i
            result_df = self.process_molecular_formula(
                molecular_formula=formula,
                smiles_index=current_smiles_index,
            )
            results_dict[formula] = result_df

        return results_dict

    def find_best_matches(
        self,
        results_df: pd.DataFrame,
        similarity_threshold: float = 0.8,
        max_results: int = 10,
        sort_by: str = "similarity",
    ) -> pd.DataFrame:
        """
        Find the best matching structures from results.

        Args:
            results_df: DataFrame with similarity results from `process_molecular_formula`.
            similarity_threshold: Minimum 'similarity' (weighted) threshold.
            max_results: Maximum number of results to return.
            sort_by: Column to sort by. Common choices: "similarity",
                     "sum_of_all_individual_similarities", "ir_similarity",
                     "cnmr_similarity", "hnmr_similarity", or "tanimoto".

        Returns:
            DataFrame with best matches.
        """
        if results_df.empty:
            logger.warning("Input results_df is empty. Returning empty DataFrame.")
            return pd.DataFrame()

        if "similarity" not in results_df.columns:
            # Fallback: try to sort by 'sort_by' if it exists, otherwise return head
            if sort_by in results_df.columns:
                return results_df.sort_values(by=sort_by, ascending=False).head(max_results)
            return results_df.head(max_results)

        if sort_by not in results_df.columns:
            logger.warning(
                f"Sort column '{sort_by}' not found in results. Defaulting to 'similarity'. "
                f"Available columns: {results_df.columns.tolist()}"
            )
            sort_by = "similarity"  # 'similarity' is confirmed to exist by now or errored above

        filtered_df = results_df[results_df["similarity"] >= similarity_threshold]

        if filtered_df.empty:
            logger.info(f"No results found above similarity threshold {similarity_threshold}.")
            return pd.DataFrame()

        sorted_df = filtered_df.sort_values(by=sort_by, ascending=False)
        return sorted_df.head(max_results)

    def _filter_by_molecular_formulas(self, df: Any, molecular_formulas: list[str]) -> Any:
        """Helper method to filter Polars or Pandas DataFrame by molecular formulas."""
        if df is None:
            logger.warning("DataFrame for filtering is None.")
            # Depending on how Polars handles None, this might need to return an empty Polars DF
            return None  # Or an empty Polars DataFrame: polars.DataFrame()

        # Check for Polars DataFrame (duck typing)
        if hasattr(df, "lazy") and callable(df.lazy) and hasattr(df, "filter"):
            return df.filter(df["molecular_formula"].is_in(molecular_formulas))
        elif isinstance(df, pd.DataFrame):
            return df[df["molecular_formula"].isin(molecular_formulas)]
        return df


# Example usage (adjust paths as needed)
if __name__ == "__main__":
    # Create analyzer, e.g., only use IR and CNMR for weighted similarity
    analyzer = SimpleMoleculeAnalyzer(
        models_config_path="../../configs",  # Adjust if your configs are elsewhere
        results_dir="./analysis_results_custom",
        cache_dir="./cache_custom",
        active_spectra=["ir", "cnmr", "hnmr"],
    )

    analyzer.load_models()
    analyzer.load_data(
        dataset_path="../valid_dataset_20250519_1151.pkl",
        ir_embeddings_path="../7_test_set_ir_embeddings_large_dataset_20250519_1151.pkl",
        cnmr_embeddings_path="../7_test_set_cnmr_embeddings_large_dataset_20250519_1150.pkl",
        hnmr_embeddings_path="../7_hnmr_simulated_detailed_cnn_architecture_large_dataset_20250519_1149.pkl",
    )

    results_df = analyzer.process_molecular_formula(smiles_index=3)
