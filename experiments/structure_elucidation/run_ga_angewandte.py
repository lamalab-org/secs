import json
import pickle as pkl
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gafuncs import CachedBatchFunction
from loguru import logger
from mol_ga.graph_ga.gen_candidates import graph_ga_blended_generation
from mol_ga.preconfigured_gas import default_ga
from prune_sim import read_weights, tanimoto_similarity, tokenize_string
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from molbind.data.analysis import aggregate_embeddings


def encode_smiles(individual, cnmr_model, hnmr_model):
    input_ids, attention_mask = tokenize_string(individual, "smiles")
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    with torch.inference_mode():
        cnmr_embedding = cnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")
        hnmr_embedding = hnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")

    return cnmr_embedding, hnmr_embedding


def gpu_encode_smiles(individuals, cnmr_model, hnmr_model):
    # split into chunks of 128
    chunk_size = 8192
    cnmr_embeddings = []
    hnmr_embeddings = []
    chunk_range = [*list(range(0, len(individuals), chunk_size)), len(individuals)]
    for i, j in enumerate(tqdm(chunk_range[:-1])):
        cnmr_embedding, hnmr_embedding = encode_smiles(
            individuals[j : chunk_range[i + 1]],
            cnmr_model,
            hnmr_model,
        )
        torch.cuda.empty_cache()
        cnmr_embeddings.append(cnmr_embedding)
        hnmr_embeddings.append(hnmr_embedding)
    cnmr_embeddings = torch.cat(cnmr_embeddings, dim=0)
    hnmr_embeddings = torch.cat(hnmr_embeddings, dim=0)
    return cnmr_embeddings, hnmr_embeddings


def compute_individual_atom_counts(individual):
    mol = Chem.MolFromSmiles(individual)
    # add hydrogen atoms
    mol = Chem.AddHs(mol)
    if mol is None:
        return None
    atom_counts = {}
    for atom in mol.GetAtoms():
        atom_counts[atom.GetSymbol()] = atom_counts.get(atom.GetSymbol(), 0) + 1
    return atom_counts


def binary_vector_numpy(length):
    generator = np.random.default_rng(seed=42)
    return generator.integers(1, 3, size=length)


def reward_function_molecular_formula(
    individuals: list[str],
    cnmr_model: torch.nn.Module,
    hnmr_model: torch.nn.Module,
    id_spectra: int,
    atom_counts_original_smiles: dict,
    cnmr_embeddings_path: str,
    hnmr_embeddings_path: str,
    dataset_path: str,
):
    cnmr_embedding, hnmr_embedding = gpu_encode_smiles(individuals, cnmr_model, hnmr_model)
    # original smiles embedding
    spectra_cnmr_embedding, spectra_hnmr_embedding = find_spectra_embeddings(
        id_spectra,
        cnmr_embeddings_path,
        hnmr_embeddings_path,
        dataset_path,
    )
    # count individual atom types
    molecular_formula_loss = np.zeros(len(individuals))
    nr_of_atoms_in_original_smiles = sum(atom_counts_original_smiles.values())

    for individual_idx, individual in enumerate(individuals):
        atom_counts_individual = compute_individual_atom_counts(individual)
        molecular_formula_loss_per_individual = 0
        if atom_counts_individual is None:
            continue
        # calculate the difference in atom counts
        for atom, count in atom_counts_individual.items():
            molecular_formula_loss_per_individual += abs(count - atom_counts_original_smiles.get(atom, 0))
        molecular_formula_loss[individual_idx] = -molecular_formula_loss_per_individual / nr_of_atoms_in_original_smiles
    cnmr_cosine_similarity = cosine_similarity(cnmr_embedding, spectra_cnmr_embedding, dim=1).cpu().numpy()
    hnmr_cosine_similarity = cosine_similarity(hnmr_embedding, spectra_hnmr_embedding, dim=1).cpu().numpy()
    sum_cosine_similarity = cnmr_cosine_similarity + hnmr_cosine_similarity
    return sum_cosine_similarity / 2 + molecular_formula_loss


def find_spectra_embeddings(
    index_of_smiles_to_test: int,
    cnmr_embeddings_path: str,
    hnmr_embeddings_path: str,
    dataset_path: str,
) -> None:
    index_of_smiles_to_test = int(index_of_smiles_to_test)

    c_nmr_embeddings = pd.read_pickle(cnmr_embeddings_path)
    h_nmr_embeddings = pd.read_pickle(hnmr_embeddings_path)
    # sum up spectra embeddings
    list_of_smiles = pd.read_pickle(dataset_path).smiles.to_list()
    original_smiles = list_of_smiles[index_of_smiles_to_test]
    # print molecular formula of the original smiles
    # print the original smiles
    logger.debug(f"Original smiles: {original_smiles}")
    c_nmr_embeddings = aggregate_embeddings(
        embeddings=c_nmr_embeddings,
        modalities=["smiles", "c_nmr"],
        central_modality="smiles",
    )
    h_nmr_embeddings = aggregate_embeddings(
        embeddings=h_nmr_embeddings,
        modalities=["smiles", "h_nmr_cnn"],
        central_modality="smiles",
    )
    spectra_cnmr_embedding = c_nmr_embeddings["c_nmr"][index_of_smiles_to_test].to("cuda")
    spectra_hnmr_embedding = h_nmr_embeddings["h_nmr_cnn"][index_of_smiles_to_test].to("cuda")
    return spectra_cnmr_embedding, spectra_hnmr_embedding


def load_models(configs_path: str, cnmr_experiment: str, hnmr_experiment: str):
    from hydra import compose, initialize

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
    cnmr_model = read_weights(cnmr_config)
    hnmr_model = read_weights(hnmr_config)
    return cnmr_model, hnmr_model


def calculate_molar_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return CalcExactMolWt(mol)


def main(
    result_file,
    cnmr_model,
    hnmr_model,
    rng: random.Random,
    cache_dir="cache_ga",
    init_population_size=10,
    number_of_generations=100,
):
    # load the original smiles
    file_with_results = pd.read_csv(result_file)
    spectra_id = result_file.split("/")[-1].split("_")[0]
    valid_dataset_filename = "valid_dataset_20241113_1514.pkl"
    # find the one with tanimoto=1
    # original_smiles = file_with_results[file_with_results["tanimoto"] == 1]["canonical_smiles"].to_list()[0]
    read_file_2_test_set = pd.read_pickle(f"../{valid_dataset_filename}")
    original_smiles = read_file_2_test_set.smiles.to_list()[int(spectra_id)]
    # drop original smiles
    logger.debug(len(file_with_results))
    file_with_results = file_with_results[file_with_results["tanimoto"] != 1]
    if len(file_with_results) < init_population_size:
        init_population_size = len(file_with_results)
    file_with_results = file_with_results[file_with_results["sascore"] < 7]
    top_n = (
        file_with_results.sort_values(by="sum_of_similarities", ascending=False)
        .head(init_population_size)["canonical_smiles"]
        .to_list()
    )
    init_reward = partial(
        reward_function_molecular_formula,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
        id_spectra=spectra_id,
        atom_counts_original_smiles=compute_individual_atom_counts(original_smiles),
        cnmr_embeddings_path="../cnmr_angewandte_20241113_1514.pkl",
        hnmr_embeddings_path="../hnmr_angewandte_20241113_1514.pkl",
        dataset_path=f"../{valid_dataset_filename}",
    )

    logger.debug(len(file_with_results))
    ga_results = default_ga(
        starting_population_smiles=top_n,
        scoring_function=CachedBatchFunction(init_reward),
        max_generations=number_of_generations,
        offspring_size=1024,
        population_size=256,
        logger=logger,
        rng=rng,
        offspring_gen_func=partial(graph_ga_blended_generation, frac_graph_ga_mutate=0.1),
        # parallel=parallel,
    )
    logger.info(f"Best individual: {max(ga_results.population)}")
    logger.info(f"Original smiles: {original_smiles}")
    # return if ga_results.population is the same as original smiles
    cache_dict = {
        "initial_population": list(top_n),
        "final_population": {v: np.float64(str(k)) for k, v in ga_results.population},
        "best_individual": max(ga_results.population)[1],
        "best_score": np.float64(max(ga_results.population)[0]),
        "original_smiles": original_smiles,
        "tanimoto_similarity": np.float64(tanimoto_similarity(max(ga_results.population)[1], original_smiles)),
    }
    # save to json
    with Path(f"{cache_dir}/{spectra_id}.json").open("w") as f:
        json.dump(cache_dict, f, indent=4)
    # save info to pickle
    with Path(f"{cache_dir}/{spectra_id}.pkl").open("wb") as f:
        pkl.dump(ga_results.gen_info, f, protocol=pkl.HIGHEST_PROTOCOL)
    return ga_results


def cli(
    start_range: int,
    end_range: int,
    seed: int,
    cache_dir="angewandte",
    init_population_size=512,
    cnmr_experiment="metrics/cnmr_simulated_large_dataset_testset",
    hnmr_experiment="metrics/hnmr_cnn_detailed_large_dataset",
    generations: int = 50,
):
    start_range = int(start_range)
    end_range = int(end_range)
    init_population_size = int(init_population_size)
    cnmr_model, hnmr_model = load_models(
        "../../configs",
        cnmr_experiment=cnmr_experiment,
        hnmr_experiment=hnmr_experiment,
    )
    # check if cache_dir exists
    Path(f"{cache_dir}_{init_population_size}_seed_{seed}").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for i in range(start_range, end_range):
        del rng
        rng = random.Random(seed)
        rng.seed(seed)
        try:
            main(
                f"results_angewandte/{i}_out_sim.csv",
                cnmr_model,
                hnmr_model,
                init_population_size=init_population_size,
                cache_dir=f"{cache_dir}_{init_population_size}_seed_{seed}",
                rng=rng,
                number_of_generations=generations,
            )
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


if __name__ == "__main__":
    import fire

    # lightning seed experiment
    from pytorch_lightning import seed_everything
    seed_everything(42)
    fire.Fire(cli)
