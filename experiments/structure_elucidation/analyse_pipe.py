import json
from pathlib import Path

import fire
import pandas as pd
import tqdm
from joblib import Parallel, delayed
from loguru import logger
from prune_sim import tanimoto_similarity
from rdkit import Chem
from rdkit.rdBase import BlockLogs

from molbind.utils.spec2struct import smiles_to_molecular_formula


def get_jsons(path: Path) -> list[Path]:
    """Get all JSON files in the given path."""
    return list(Path().rglob(f"{path}*/idx_*/*.json"))


def get_pickles(path: Path) -> list[Path]:
    """Get all Pickle files in the given path."""
    return list(Path().rglob(f"{path}*/idx_*/*.pkl"))


def smiles_is_radical_or_is_charged_or_has_wrong_valence(smiles: str) -> bool:
    """
    Determines if a SMILES string represents a radical, charged molecule, or has wrong valence.

    Args:
        smiles (str): SMILES string representation of a molecule

    Returns:
        bool: True if the molecule is a radical OR charged OR has wrong valence, False otherwise
    """
    try:
        # Parse the SMILES string into a molecule object - without sanitization first
        mol = Chem.MolFromSmiles(smiles, sanitize=False)

        # Return False if SMILES is invalid
        if mol is None:
            return False

        # Check 1: Overall charge neutrality (before adding hydrogens)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if total_charge != 0:
            return True  # Molecule is charged

        # Check 2: Valence validity - try to sanitize
        try:
            # This will raise an exception if valence is invalid
            Chem.SanitizeMol(mol)
        except Exception:
            return True  # Molecule has wrong valence

        # Add hydrogens after sanitization succeeds
        mol = Chem.AddHs(mol)

        # Check 3: Unpaired electrons (radicals)
        for atom in mol.GetAtoms():
            # Get the number of radical electrons (unpaired electrons)
            num_radical_electrons = atom.GetNumRadicalElectrons()

            # If any atom has unpaired electrons, it's a radical
            if num_radical_electrons > 0:
                return True  # Molecule is a radical

        return False  # Molecule is neutral, has valid valence, and no radicals

    except Exception:
        # Return True for any parsing errors (likely invalid structures)
        return True


def get_data_from_files(summary_file: Path, details_file: Path):
    """Reads summary and details files, returning processed molecular data."""
    pkl_file = pd.read_pickle(details_file)
    with Path(summary_file).open() as f:
        summary = json.load(f)
    original_smiles = summary["original_smiles"]
    original_formula = smiles_to_molecular_formula(original_smiles)
    population = {v: k for k, v in pkl_file.population}
    filtered_population = {
        v: k
        for k, v in pkl_file.population
        if smiles_to_molecular_formula(v) == original_formula
        and "." not in v
        and not smiles_is_radical_or_is_charged_or_has_wrong_valence(v)
    }

    return population, filtered_population, original_formula, original_smiles


def calculate_tanimoto_scores(file_info_tuple):
    """
    Calculates Tanimoto similarities for a given row's data.
    The input tuple is the return value of get_data_from_files.
    """
    _population, filtered_population, _original_formula, original_smiles = file_info_tuple
    # The keys of filtered_population are the SMILES strings to compare against.
    return [tanimoto_similarity(original_smiles, k) for k in filtered_population]


def check_top_k(score_list, k):
    """Checks if the target score (1.0) is within the top k results."""
    if not score_list:
        return 0.0
    # The original SMILES has a Tanimoto similarity of 1.0 with itself and should be in the top k scores.
    if 1.0 in score_list[:k]:
        return 1.0
    return 0.0


def analyse_pipe(path: Path, n_jobs: int = -1):
    """
    Analyzes the output files in a parallelized manner.

    Args:
        path (Path): The root directory to search for files.
        n_jobs (int): The number of CPU cores to use. -1 means use all available cores.
    """
    logger.info(f"Using {n_jobs if n_jobs > 0 else 'all available'} CPU cores for parallel processing.")

    get_all_jsons = get_jsons(path)
    get_pickle = get_pickles(path)
    indices = [int(json_file.stem.split("_")[0]) for json_file in get_all_jsons]
    dataframe = pd.DataFrame({"idx": indices, "summary": get_all_jsons, "details": get_pickle})

    # 2. Parallelize the file reading and initial processing.
    # This replaces the slow `progress_apply` with a parallel map operation.
    # Each core will process a different file pair simultaneously.
    logger.info(f"Step 1/4: Reading and processing {len(dataframe)} file pairs in parallel...")
    file_info_list = Parallel(n_jobs=n_jobs)(
        delayed(get_data_from_files)(row.summary, row.details)
        for row in tqdm.tqdm(dataframe.itertuples(), total=len(dataframe), desc="Processing files")
    )
    dataframe["file_info"] = file_info_list

    # 3. Parallelize the Tanimoto score calculation.
    # This is a CPU-bound task that benefits greatly from parallelization.
    logger.info("Step 2/4: Calculating Tanimoto scores in parallel...")
    scores_list = Parallel(n_jobs=n_jobs)(
        delayed(calculate_tanimoto_scores)(info) for info in tqdm.tqdm(dataframe["file_info"], desc="Calculating scores")
    )
    dataframe["score"] = scores_list

    # --- This section remains sequential as it involves merging data, ---
    # --- which is complex to parallelize safely and is usually fast enough. ---
    logger.info("Step 3/4: Merging population data...")
    dataframe["full_population"] = dataframe["file_info"].apply(lambda x: x[0])
    merged_population = {}
    for idx, group in dataframe.groupby("idx"):
        merged_population[idx] = {}
        for population in group["full_population"]:
            for smiles, score in population.items():
                if smiles not in merged_population[idx] or merged_population[idx][smiles] < score:
                    merged_population[idx][smiles] = score
    dataframe["merged_population"] = dataframe["idx"].map(merged_population)
    # --- End of sequential section ---

    # 4. Parallelize the Top-K analysis.
    # While `check_top_k` is fast, parallelizing this can still be quicker for very large dataframes.
    logger.info("Step 4/4: Calculating Top-K metrics in parallel...")
    with tqdm.tqdm(total=3, desc="Calculating Top-K") as pbar:
        dataframe["top1"] = Parallel(n_jobs=n_jobs)(delayed(check_top_k)(s, 1) for s in dataframe["score"])
        pbar.update(1)
        dataframe["top5"] = Parallel(n_jobs=n_jobs)(delayed(check_top_k)(s, 5) for s in dataframe["score"])
        pbar.update(1)
        dataframe["top10"] = Parallel(n_jobs=n_jobs)(delayed(check_top_k)(s, 10) for s in dataframe["score"])
        pbar.update(1)
        dataframe["top20"] = Parallel(n_jobs=n_jobs)(delayed(check_top_k)(s, 20) for s in dataframe["score"])
        pbar.update(1)

    output_file = f"results_{path}.pkl"
    dataframe.to_pickle(output_file)
    logger.info(f"Analysis complete. Results saved to {output_file}")

    # Calculate and print final stats
    total_rows = len(dataframe)
    if total_rows > 0:
        logger.info(f"Top 1 Accuracy: {dataframe['top1'].sum() / total_rows:.2%}")
        logger.info(f"Top 5 Accuracy: {dataframe['top5'].sum() / total_rows:.2%}")
        logger.info(f"Top 10 Accuracy: {dataframe['top10'].sum() / total_rows:.2%}")
        logger.info(f"Top 20 Accuracy: {dataframe['top20'].sum() / total_rows:.2%}")
    else:
        logger.warning("No data was processed.")


if __name__ == "__main__":
    BlockLogs()
    logger.info("Starting analysis...")
    fire.Fire(analyse_pipe)
