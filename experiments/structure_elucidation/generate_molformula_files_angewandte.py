import re
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd
from gafuncs import sascore
from loguru import logger
from maygen_out_to_canonical import isomer_to_canonical
from prune_sim_angewandte import embedding_pruning, get_number_of_topologically_distinct_atoms, load_models, tanimoto_similarity
from request_pubchem import (
    convert_to_molecular_formulas,
    read_cached_CID_smiles,
)

from molbind.data.analysis.utils import aggregate_embeddings
from molbind.models import MolBind


def main(
    file_with_isomers: str,
    pruned_file: str,
    index_of_smiles_to_test: int,
    dataset_path: str,
    cnmr_embeddings_path: str,
    hnmr_embeddings_path: str,
    cnmr_model: MolBind,
    hnmr_model: MolBind,
    folder_results: str,
    cnmr_ratio: float = 1,
    hnmr_ratio: float = 1,
    synthetic_access_quantile: float | None = None,
) -> None:
    index_of_smiles_to_test = int(index_of_smiles_to_test)
    c_nmr_embeddings = pd.read_pickle(cnmr_embeddings_path)
    h_nmr_embeddings = pd.read_pickle(hnmr_embeddings_path)
    # sum up spectra embeddings
    list_of_smiles = pd.read_pickle(dataset_path).smiles.to_list()
    original_smiles = list_of_smiles[index_of_smiles_to_test]
    # print molecular formula of the original smiles
    # print the original smilesa
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

    isomer_df = pd.read_csv(file_with_isomers)
    # drop duplicates
    isomer_df = isomer_df.drop_duplicates(subset=["canonical_smiles"])
    isomer_df["unique_hydrogens"] = isomer_df["canonical_smiles"].progress_apply(
        lambda x: get_number_of_topologically_distinct_atoms(x, atomic_number=1)
    )
    isomer_df["unique_carbons"] = isomer_df["canonical_smiles"].progress_apply(
        lambda x: get_number_of_topologically_distinct_atoms(x, atomic_number=6)
    )
    isomer_df["sascore"] = isomer_df["canonical_smiles"].progress_apply(sascore)
    if synthetic_access_quantile:
        logger.info("You requested to filter based on synthetic accessibility")
        isomer_df = isomer_df[isomer_df["synthetic_access"] < isomer_df["synthetic_access"].quantile(synthetic_access_quantile)]
        logger.debug(f"Length of isomer_df after synthetic access filtering: {len(isomer_df)}")

    logger.debug(f"Length of isomer_df: {len(isomer_df)}")
    (
        cosine_similarities,
        cnmr_similarities,
        hnmr_similarities,
        cnmr_hnmr_similarities,
    ) = embedding_pruning(
        spectra_cnmr_embedding=spectra_cnmr_embedding,
        spectra_hnmr_embedding=spectra_hnmr_embedding,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
        smiles=isomer_df["canonical_smiles"].to_list(),
        cnmr_ratio=cnmr_ratio,
        hnmr_ratio=hnmr_ratio,
    )
    cosine_similarities = cosine_similarities.tolist()
    cnmr_similarities = cnmr_similarities.tolist()
    hnmr_similarities = hnmr_similarities.tolist()
    cnmr_hnmr_similarities = cnmr_hnmr_similarities.tolist()
    isomer_df["cnmr_similarity"] = cnmr_similarities
    isomer_df["hnmr_similarity"] = hnmr_similarities
    isomer_df["similarity"] = cosine_similarities
    isomer_df["sum_of_similarities"] = isomer_df["cnmr_similarity"] + isomer_df["hnmr_similarity"]
    isomer_df["cnmr_hnmr_similarity"] = cnmr_hnmr_similarities

    # tanimoto similarity with the original smiles
    tanimoto = partial(tanimoto_similarity, original_smiles)
    isomer_df["tanimoto"] = isomer_df["canonical_smiles"].progress_apply(lambda x: tanimoto(x))
    # save backup
    csv_path = Path(folder_results) / Path(
        str(index_of_smiles_to_test) + "_" + str(Path(pruned_file).with_suffix("")) + "_sim"
    ).with_suffix(".csv")

    isomer_df.to_csv(csv_path, index=False)


def run_scripts_pipe(
    molecular_formula,
    smiles_index,
    cnmr_model,
    hnmr_model,
    pubchem_cache=None,
    cnmr_embeddings_path: str | None = None,
    hnmr_embeddings_path: str | None = None,
    dataset_path: str | None = None,
    folder_results: str | None = None,
):
    # run script 1: request_pubchem.py
    # os.system(f"python request_pubchem.py '{molecular_formula}'")
    # read cached CID_smiles
    # filter data based on molecular formula
    # pubchem_cache = pubchem_cache.filter(
    #     pl.col("molecular_formula") == molecular_formula
    # )
    # save smiles to a file
    # if not Path(f"isomers/{molecular_formula}.txt").exists():
    with Path(f"isomers_multiform/{molecular_formula}.txt").open("w") as f:
        for smiles in pubchem_cache["smiles"]:
            f.write(smiles + "\n")
    cid_file = f"isomers_multiform/{molecular_formula}.txt"
    # cids is a txt file with a list of CIDs
    with Path(cid_file).open("r") as f:
        smiles = f.read().splitlines()
    logger.info(f"Found {len(smiles)} CIDs for molecular formula {molecular_formula}")
    # run script 2: maygen_out_to_canonical.py (converts SMILES to canonical SMILES)
    isomer_to_canonical(cid_file, f"isomers_multiform/{molecular_formula}.csv")

    main(
        f"isomers_multiform/{molecular_formula}.csv",
        "out.csv",
        index_of_smiles_to_test=smiles_index,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
        cnmr_embeddings_path=cnmr_embeddings_path,
        hnmr_embeddings_path=hnmr_embeddings_path,
        dataset_path=dataset_path,
        folder_results=folder_results,
    )


# Function to extract atom counts from a molecular formula
def get_atom_counts_from_formula(formula):
    # Regular expression to match elements and their counts
    pattern = r"([A-Z][a-z]*)(\d*)"

    # Find all matches in the formula
    matches = re.findall(pattern, formula)

    # Create a Counter to store atom counts
    atom_counts = Counter()

    # Loop through matches and update the counter
    for element, count in matches:
        count_atom = 1 if count == "" else int(count)
        atom_counts[element] += count_atom

    return dict(atom_counts)


def gen_close_molformulas_from_seed(seed_formula):
    atom_counts = get_atom_counts_from_formula(seed_formula)
    carbons = atom_counts.get("C", 0)
    hydrogens = atom_counts.get("H", 0)
    nitrogens = atom_counts.get("N", 0)
    chlorine = atom_counts.get("Cl", 0)
    bromine = atom_counts.get("Br", 0)
    fluorine = atom_counts.get("F", 0)
    phosphorus = atom_counts.get("P", 0)
    sulphur = atom_counts.get("S", 0)
    total_halogens = chlorine + bromine + fluorine
    seed_formula_0 = seed_formula.replace(f"C{carbons}", f"C{carbons - 3}").replace(f"H{hydrogens}", f"H{hydrogens - 6}")
    seed_formula_1 = seed_formula.replace(f"C{carbons}", f"C{carbons + 1}").replace(f"H{hydrogens}", f"H{hydrogens + 2}")
    seed_formula_2 = seed_formula.replace(f"C{carbons}", f"C{carbons - 1}").replace(f"H{hydrogens}", f"H{hydrogens - 2}")
    seed_formula_3 = seed_formula.replace(f"C{carbons}", f"C{carbons + 2}").replace(f"H{hydrogens}", f"H{hydrogens + 4}")
    seed_formula_4 = seed_formula.replace(f"C{carbons}", f"C{carbons - 2}").replace(f"H{hydrogens}", f"H{hydrogens - 4}")
    seed_formula_5 = seed_formula.replace(f"N{nitrogens}", f"N{nitrogens + 1}").replace(f"H{hydrogens}", f"H{hydrogens + 1}")
    seed_formula_6 = seed_formula.replace(f"N{nitrogens}", f"N{nitrogens - 1}").replace(f"H{hydrogens}", f"H{hydrogens - 1}")
    seed_formula_7 = seed_formula.replace(f"Cl{chlorine}", f"Cl{chlorine + 1}").replace(f"H{hydrogens}", f"H{hydrogens + 1}")
    seed_formula_8 = seed_formula.replace(f"Cl{chlorine}", f"Cl{chlorine - 1}").replace(f"H{hydrogens}", f"H{hydrogens - 1}")
    seed_formula_9 = seed_formula.replace(f"Br{bromine}", f"Br{bromine + 1}").replace(f"H{hydrogens}", f"H{hydrogens + 1}")
    seed_formula_10 = seed_formula.replace(f"Br{bromine}", f"Br{bromine - 1}").replace(f"H{hydrogens}", f"H{hydrogens - 1}")
    seed_formula_11 = seed_formula.replace(f"F{fluorine}", f"F{fluorine + 1}").replace(f"H{hydrogens}", f"H{hydrogens + 1}")
    # keep just C,N,H,O
    # if count is 1, then replace the element with an empty string
    seed_formula_12 = seed_formula.replace("Cl", "") if chlorine == 1 else seed_formula.replace(f"Cl{chlorine}", "")
    seed_formula_12 = seed_formula_12.replace("Br", "") if bromine == 1 else seed_formula_12.replace(f"Br{bromine}", "")
    seed_formula_12 = seed_formula_12.replace("F", "") if fluorine == 1 else seed_formula_12.replace(f"F{fluorine}", "")
    seed_formula_12 = seed_formula_12.replace(f"H{hydrogens}", f"H{hydrogens + total_halogens}")
    seed_formula_13 = seed_formula.replace("P", "") if phosphorus == 1 else seed_formula.replace(f"P{phosphorus}", "")
    seed_formula_13 = seed_formula_13.replace(f"H{hydrogens}", f"H{hydrogens + 5}")
    seed_formula_14 = seed_formula.replace("S", "") if sulphur == 1 else seed_formula.replace(f"S{sulphur}", "")
    seed_formula_14 = seed_formula_14.replace(f"H{hydrogens}", f"H{hydrogens + 4}")
    seed_formula_15 = seed_formula.replace(f"S{sulphur}", f"S{sulphur + 1}")
    seed_formula_16 = seed_formula.replace(f"S{sulphur}", f"S{sulphur - 1}")
    return [
        seed_formula_0,
        seed_formula_1,
        seed_formula_2,
        seed_formula_3,
        seed_formula_4,
        seed_formula_5,
        seed_formula_6,
        seed_formula_7,
        seed_formula_8,
        seed_formula_9,
        seed_formula_10,
        seed_formula_11,
        seed_formula_12,
        seed_formula_13,
        seed_formula_14,
        seed_formula_15,
        seed_formula_16,
    ]


def filter_polars_col_by_molecular_formulas(df, molecular_formulas):
    return df[df["molecular_formula"].isin(molecular_formulas)]


def cli(
    start_index: int,
    end_index: int,
    dataset_path: str,
    cnmr_embeddings_path: str,
    hnmr_embeddings_path: str,
    cnmr_experiment: str,
    hnmr_experiment: str,
    folder_results: str,
) -> None:
    if not Path(folder_results).exists():
        Path(folder_results).mkdir()

    cached_data = read_cached_CID_smiles().to_pandas()
    cached_data = cached_data.dropna()
    # # save the cache with molecular formula
    # cached_data.to_csv("CID-SMILES.csv", separator=",", write_header=True)
    list_of_molecular_formulas = convert_to_molecular_formulas(dataset_path)
    indexes = list(range(len(list_of_molecular_formulas)))
    # save the first 30 molecular formulas to file
    with Path("isomers_multiform/molecular_formulas.txt").open("w") as f:
        f.write("\n".join(list_of_molecular_formulas[start_index:end_index]))

    cnmr_model, hnmr_model = load_models(
        "../../configs",
        cnmr_experiment=cnmr_experiment,
        hnmr_experiment=hnmr_experiment,
    )

    for molecular_formula, i in zip(
        list_of_molecular_formulas[start_index:end_index],
        indexes[start_index:end_index],
    ):
        get_molformulas = gen_close_molformulas_from_seed(molecular_formula)
        molecular_formulas = [molecular_formula, *get_molformulas]
        filter_cached = filter_polars_col_by_molecular_formulas(cached_data, molecular_formulas)
        # add molecular formula to the cache
        logger.info(f"Processing {molecular_formula}")
        run_scripts_pipe(
            molecular_formula,
            i,
            pubchem_cache=filter_cached,
            cnmr_model=cnmr_model,
            hnmr_model=hnmr_model,
            dataset_path=dataset_path,
            cnmr_embeddings_path=cnmr_embeddings_path,
            hnmr_embeddings_path=hnmr_embeddings_path,
            folder_results=folder_results,
        )
        logger.info(f"Finished processing {molecular_formula}")


if __name__ == "__main__":
    cli(
        start_index=0,
        end_index=5,
        dataset_path="../valid_dataset_20241110_1616.pkl",
        cnmr_embeddings_path="../cnmr_angewandte_20241110_1614.pkl",
        hnmr_embeddings_path="../hnmr_angewandte_20241110_1616.pkl",
        cnmr_experiment="metrics/cnmr_simulated_large_dataset_testset",
        hnmr_experiment="metrics/hnmr_cnn_detailed_large_dataset",
        folder_results="results_angewandte",
    )
