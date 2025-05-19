import json
import pickle as pkl
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gafuncs import CachedBatchFunction  # Assuming gafuncs is a custom module
from loguru import logger
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
from mol_ga.preconfigured_gas import default_ga
from prune_sim import read_weights, tanimoto_similarity, tokenize_string  # Assuming prune_sim is a custom module
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from molbind.data.analysis import aggregate_embeddings

# Assuming molbind.utils.spec2struct is available for smiles_to_molecular_formula
from molbind.utils.spec2struct import smiles_to_molecular_formula


def encode_smiles(individual, ir_model, cnmr_model, hnmr_model):
    input_ids, attention_mask = tokenize_string(individual, "smiles")
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    with torch.inference_mode():
        ir_embedding = ir_model.encode_modality((input_ids, attention_mask), modality="smiles")
        cnmr_embedding = cnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")
        hnmr_embedding = hnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")

    return ir_embedding, cnmr_embedding, hnmr_embedding


def gpu_encode_smiles(individuals, ir_model, cnmr_model, hnmr_model):
    # split into chunks of 128
    chunk_size = 8192  # Consider making this configurable or smaller if OOM occurs
    ir_embeddings = []
    cnmr_embeddings = []
    hnmr_embeddings = []
    if not individuals:  # Handle empty list
        return torch.empty(0), torch.empty(0), torch.empty(0)

    chunk_range = [*list(range(0, len(individuals), chunk_size)), len(individuals)]
    for i, j in enumerate(tqdm(chunk_range[:-1], desc="Encoding SMILES")):
        ir_embedding, cnmr_embedding, hnmr_embedding = encode_smiles(
            individuals[j : chunk_range[i + 1]],
            ir_model,
            cnmr_model,
            hnmr_model,
        )
        torch.cuda.empty_cache()
        ir_embeddings.append(ir_embedding)
        cnmr_embeddings.append(cnmr_embedding)
        hnmr_embeddings.append(hnmr_embedding)

    if not ir_embeddings:  # Handle case where individuals length < chunk_size or empty
        return torch.empty(0, device="cuda"), torch.empty(0, device="cuda"), torch.empty(0, device="cuda")

    ir_embeddings = torch.cat(ir_embeddings, dim=0)
    cnmr_embeddings = torch.cat(cnmr_embeddings, dim=0)
    hnmr_embeddings = torch.cat(hnmr_embeddings, dim=0)
    return ir_embeddings, cnmr_embeddings, hnmr_embeddings


def compute_individual_atom_counts(individual: str):
    mol = Chem.MolFromSmiles(individual)
    if mol is None:
        logger.warning(f"Could not parse SMILES: {individual} in compute_individual_atom_counts")
        return None
    mol = Chem.AddHs(mol)  # Add hydrogens explicitly
    atom_counts = {}
    for atom in mol.GetAtoms():
        atom_counts[atom.GetSymbol()] = atom_counts.get(atom.GetSymbol(), 0) + 1
    return atom_counts


def binary_vector_numpy(length):
    generator = np.random.default_rng(seed=42)
    return generator.integers(1, 3, size=length)


def reward_function_molecular_formula(
    individuals: list[str],
    ir_model: torch.nn.Module,
    cnmr_model: torch.nn.Module,
    hnmr_model: torch.nn.Module,
    id_spectra: int,  # Changed: This will now be an integer index
    atom_counts_original_smiles: dict,
    cnmr_embeddings_path: str,
    ir_embeddings_path: str,
    hnmr_embeddings_path: str,
    dataset_path: str,
):
    if not individuals:  # Handle empty individuals list
        return np.array([])

    ir_embedding, cnmr_embedding, hnmr_embedding = gpu_encode_smiles(individuals, ir_model, cnmr_model, hnmr_model)

    spectra_ir_embedding, spectra_cnmr_embedding, spectra_hnmr_embedding = find_spectra_embeddings(
        id_spectra,  # Pass the integer index
        ir_embeddings_path,
        cnmr_embeddings_path,
        hnmr_embeddings_path,
        dataset_path,
    )

    molecular_formula_loss = np.zeros(len(individuals))
    nr_of_atoms_in_original_smiles = sum(atom_counts_original_smiles.values())
    if nr_of_atoms_in_original_smiles == 0:  # Avoid division by zero
        logger.warning("Number of atoms in original SMILES is zero.")
        # Handle this case appropriately, e.g., by setting loss to a large negative value or skipping this penalty
        nr_of_atoms_in_original_smiles = 1  # Avoid division by zero, effectively making loss 0 if counts are also 0

    for individual_idx, individual in enumerate(individuals):
        atom_counts_individual = compute_individual_atom_counts(individual)
        molecular_formula_loss_per_individual = 0
        if atom_counts_individual is None:
            # Penalize invalid SMILES heavily
            molecular_formula_loss[individual_idx] = -100  # Large penalty
            continue

        all_atoms = set(atom_counts_individual.keys()) | set(atom_counts_original_smiles.keys())
        for atom in all_atoms:
            molecular_formula_loss_per_individual += abs(
                atom_counts_individual.get(atom, 0) - atom_counts_original_smiles.get(atom, 0)
            )

        if nr_of_atoms_in_original_smiles > 0:
            molecular_formula_loss[individual_idx] = -molecular_formula_loss_per_individual / nr_of_atoms_in_original_smiles
        else:  # if original smiles had 0 atoms (edge case)
            molecular_formula_loss[individual_idx] = -molecular_formula_loss_per_individual

    ir_cosine_similarity = cosine_similarity(ir_embedding, spectra_ir_embedding.unsqueeze(0), dim=1).cpu().numpy()
    cnmr_cosine_similarity = cosine_similarity(cnmr_embedding, spectra_cnmr_embedding.unsqueeze(0), dim=1).cpu().numpy()
    hnmr_cosine_similarity = cosine_similarity(hnmr_embedding, spectra_hnmr_embedding.unsqueeze(0), dim=1).cpu().numpy()

    sum_cosine_similarity = ir_cosine_similarity + cnmr_cosine_similarity + hnmr_cosine_similarity

    return sum_cosine_similarity / 3 + molecular_formula_loss


def find_spectra_embeddings(
    index_of_smiles_to_test: int,  # Changed: This is now an integer index
    ir_embeddings_path: str,
    cnmr_embeddings_path: str,
    hnmr_embeddings_path: str,
    dataset_path: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Corrected return type
    # index_of_smiles_to_test = int(index_of_smiles_to_test) # REMOVED: No longer needed, it's already an int

    ir_embeddings_df = pd.read_pickle(ir_embeddings_path)
    c_nmr_embeddings_df = pd.read_pickle(cnmr_embeddings_path)
    h_nmr_embeddings_df = pd.read_pickle(hnmr_embeddings_path)

    dataset_df = pd.read_pickle(dataset_path)
    list_of_smiles = dataset_df.smiles.to_list()

    if not (0 <= index_of_smiles_to_test < len(list_of_smiles)):
        raise IndexError(
            f"index_of_smiles_to_test {index_of_smiles_to_test} is out of bounds for dataset size {len(list_of_smiles)}"
        )

    original_smiles = list_of_smiles[index_of_smiles_to_test]
    logger.debug(f"Original smiles for index {index_of_smiles_to_test}: {original_smiles}")

    # Assuming aggregate_embeddings returns a dict where keys are modalities
    # and values are lists/arrays of embeddings, indexed corresponding to the original dataset.
    aggregated_ir_embeddings = aggregate_embeddings(
        embeddings=ir_embeddings_df, modalities=["smiles", "ir"], central_modality="smiles"
    )
    aggregated_c_nmr_embeddings = aggregate_embeddings(
        embeddings=c_nmr_embeddings_df,
        modalities=["smiles", "c_nmr"],
        central_modality="smiles",
    )
    aggregated_h_nmr_embeddings = aggregate_embeddings(
        embeddings=h_nmr_embeddings_df,
        modalities=["smiles", "h_nmr_cnn"],
        central_modality="smiles",
    )

    spectra_ir_embedding = aggregated_ir_embeddings["ir"][index_of_smiles_to_test].to("cuda")
    spectra_cnmr_embedding = aggregated_c_nmr_embeddings["c_nmr"][index_of_smiles_to_test].to("cuda")
    spectra_hnmr_embedding = aggregated_h_nmr_embeddings["h_nmr_cnn"][index_of_smiles_to_test].to("cuda")

    return spectra_ir_embedding, spectra_cnmr_embedding, spectra_hnmr_embedding


def load_models(configs_path: str, ir_experiment: str, cnmr_experiment: str, hnmr_experiment: str):
    from hydra import compose, initialize

    with initialize(version_base="1.3", config_path=configs_path):
        ir_config = compose(config_name="molbind_config", overrides=[f"experiment={ir_experiment}"])
    with initialize(version_base="1.3", config_path=configs_path):
        hnmr_config = compose(
            config_name="molbind_config",
            overrides=[f"experiment={hnmr_experiment}"],
        )
    with initialize(version_base="1.3", config_path=configs_path):
        cnmr_config = compose(
            config_name="molbind_config",
            overrides=[f"experiment={cnmr_experiment}"],
        )
    ir_model = read_weights(ir_config)
    cnmr_model = read_weights(cnmr_config)
    hnmr_model = read_weights(hnmr_config)
    return ir_model, cnmr_model, hnmr_model


def calculate_molar_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0  # Handle invalid SMILES
    return CalcExactMolWt(mol)


def main(
    result_df: pd.DataFrame,
    ir_model: torch.nn.Module,
    cnmr_model: torch.nn.Module,
    hnmr_model: torch.nn.Module,
    rng: random.Random,
    target_spectra_idx: int,  # Added: integer index of the target spectra
    cache_dir="cache_ga",
    init_population_size=512,
    number_of_generations=20,
):
    read_file_2_test_set = pd.read_pickle("../valid_dataset_20250519_1151.pkl")
    all_smiles_list = read_file_2_test_set.smiles.to_list()

    if not (0 <= target_spectra_idx < len(all_smiles_list)):
        logger.error(f"Target spectra index {target_spectra_idx} is out of bounds. Skipping.")
        return None

    original_smiles = all_smiles_list[target_spectra_idx]
    logger.info(f"Target original SMILES for GA (index {target_spectra_idx}): {original_smiles}")

    # Filter out the original smiles from the initial population if it's present
    file_with_results = result_df[result_df["canonical_smiles"] != original_smiles]
    # The original code used tanimoto == 1, which might be more robust if canonical_smiles can vary.
    file_with_results = file_with_results[file_with_results["tanimoto"] != 1]

    current_init_pop_size = len(file_with_results) if len(file_with_results) < init_population_size else init_population_size

    top_n = (
        file_with_results.sort_values(by="sum_of_all_individual_similarities", ascending=False)
        .head(current_init_pop_size)["canonical_smiles"]
        .to_list()
    )

    atom_counts_orig = compute_individual_atom_counts(original_smiles)
    init_reward = partial(
        reward_function_molecular_formula,
        ir_model=ir_model,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
        id_spectra=target_spectra_idx,
        atom_counts_original_smiles=atom_counts_orig,
        ir_embeddings_path="../7_test_set_ir_embeddings_large_dataset_20250519_1151.pkl",
        cnmr_embeddings_path="../7_test_set_cnmr_embeddings_large_dataset_20250519_1150.pkl",
        hnmr_embeddings_path="../7_hnmr_simulated_detailed_cnn_architecture_large_dataset_20250519_1149.pkl",
        dataset_path="../valid_dataset_20250519_1151.pkl",
    )

    ga_results = default_ga(
        starting_population_smiles=top_n,
        scoring_function=CachedBatchFunction(init_reward),  # Ensure CachedBatchFunction is compatible
        max_generations=number_of_generations,
        offspring_size=2048,  # Consider making these configurable
        population_size=2048,
        logger=logger,
        rng=rng,
        offspring_gen_func=partial(graph_ga_blended_generation, frac_graph_ga_mutate=0.1),
    )

    best_score, best_individual_smiles = max(ga_results.population)

    logger.info(f"Best individual: {best_individual_smiles} with score {best_score}")
    logger.info(f"Original smiles: {original_smiles}")

    cache_dict = {
        "initial_population": list(top_n),
        "final_population": {v: np.float64(k) for k, v in ga_results.population},  # Store as SMILES: score
        "best_individual_smiles": best_individual_smiles,
        "best_score": np.float64(best_score),
        "original_smiles": original_smiles,
        "tanimoto_similarity_to_original": np.float64(tanimoto_similarity(best_individual_smiles, original_smiles))
        if best_individual_smiles != "N/A"
        else 0.0,
    }

    output_path = Path(cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / f"{target_spectra_idx}.json").open("w") as f:
        json.dump(cache_dict, f, indent=4)
    with (output_path / f"{target_spectra_idx}.pkl").open("wb") as f:
        pkl.dump(ga_results.gen_info, f, protocol=pkl.HIGHEST_PROTOCOL)

    return ga_results


def query_enamine_vector_db(vector_db, vector, top_k=1):
    return vector_db.search(collection_name="enamine_db_last", query_vector=vector, limit=top_k)


def cli(
    start_range: int,
    end_range: int,
    seed: int,
    cache_dir_base="FINAL_RESULTS",
    init_population_size=512,
    ir_experiment="metrics/ir_simulated_large_dataset_testset",
    cnmr_experiment="metrics/cnmr_simulated_large_dataset_testset",
    hnmr_experiment="metrics/hnmr_cnn_detailed_large_dataset_testset",
    generations: int = 20,
):
    start_range = int(start_range)
    end_range = int(end_range)
    init_population_size = int(init_population_size)

    ir_model, cnmr_model, hnmr_model = load_models(
        "../../configs",  # Relative path, ensure it's correct from execution location
        ir_experiment=ir_experiment,
        cnmr_experiment=cnmr_experiment,
        hnmr_experiment=hnmr_experiment,
    )

    # Construct the specific cache directory for this run
    current_run_cache_dir = Path(f"{cache_dir_base}_{init_population_size}_seed_{seed}")
    current_run_cache_dir.mkdir(parents=True, exist_ok=True)

    read_file_2_test_set = pd.read_pickle("../valid_dataset_20250519_1151.pkl")
    all_smiles_in_dataset = read_file_2_test_set.smiles.to_list()

    # Log molecular formulas for the range being processed
    formulas_in_range_for_logging = []
    for k_idx in range(start_range, end_range):
        smiles_for_log = all_smiles_in_dataset[k_idx]
        formulas_in_range_for_logging.append(smiles_to_molecular_formula(smiles_for_log))

    for i in range(start_range, end_range):
        results_df = analyzer.process_molecular_formula(smiles_index=i)
        if not (0 <= i < len(all_smiles_in_dataset)):
            logger.error(f"Index {i} is out of bounds for dataset size {len(all_smiles_in_dataset)}. Skipping.")
            continue

        iter_rng = random.Random(seed)

        main(
            result_df=results_df,
            ir_model=ir_model,
            cnmr_model=cnmr_model,
            hnmr_model=hnmr_model,
            rng=iter_rng,
            target_spectra_idx=i,  # Pass the integer index 'i'
            init_population_size=init_population_size,
            cache_dir=str(current_run_cache_dir),  # Pass the constructed cache_dir
            number_of_generations=generations,
        )


if __name__ == "__main__":
    import fire
    from generate_molformula_files import SimpleMoleculeAnalyzer

    # deactivate batch normalization in the models
    torch.backends.cudnn.benchmark = True
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
    fire.Fire(cli)
