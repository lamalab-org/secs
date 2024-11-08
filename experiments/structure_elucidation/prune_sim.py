import contextlib
from functools import cache, partial
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import rootutils
import torch
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Contrib.SA_Score import sascorer
from torch import Tensor
from tqdm import tqdm

from molbind.data.analysis.utils import aggregate_embeddings
from molbind.data.available import ModalityConstants
from molbind.models import MolBind
from molbind.utils import rename_keys_with_prefix

torch.manual_seed(42)
tqdm.pandas()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def read_weights(config):
    model = MolBind(config).to("cuda")
    model.load_state_dict(
        rename_keys_with_prefix(torch.load(config.ckpt_path, map_location=torch.device("cuda"))["state_dict"]),
        strict=True,
    )
    model.eval()
    return model


def tokenize_string(
    smiles: list[str],
    modality: str,
) -> tuple[Tensor, Tensor]:
    tokenized_data = ModalityConstants[modality].tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=128,
    )
    return tokenized_data["input_ids"], tokenized_data["attention_mask"]


def encode_smiles(individual, ir_model, cnmr_model, hnmr_model):
    input_ids, attention_mask = tokenize_string(individual, "smiles")
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    with torch.no_grad():
        ir_embedding = ir_model.encode_modality((input_ids, attention_mask), modality="smiles")
        cnmr_embedding = cnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")
        hnmr_embedding = hnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")

    return ir_embedding, cnmr_embedding, hnmr_embedding


def gpu_encode_smiles(individual, ir_model, cnmr_model, hnmr_model):
    # split into chunks of 128
    chunk_size = 512
    ir_embeddings = []
    cnmr_embeddings = []
    hnmr_embeddings = []
    chunk_range = [*list(range(0, len(individual), chunk_size)), len(individual)]
    for i, j in enumerate(tqdm(chunk_range[:-1])):
        ir_embedding, cnmr_embedding, hnmr_embedding = encode_smiles(
            individual[j : chunk_range[i + 1]],
            ir_model,
            cnmr_model,
            hnmr_model,
        )
        torch.cuda.empty_cache()
        ir_embeddings.append(ir_embedding)
        cnmr_embeddings.append(cnmr_embedding)
        hnmr_embeddings.append(hnmr_embedding)
    ir_embeddings = torch.cat(ir_embeddings, dim=0)
    cnmr_embeddings = torch.cat(cnmr_embeddings, dim=0)
    hnmr_embeddings = torch.cat(hnmr_embeddings, dim=0)
    return ir_embeddings, cnmr_embeddings, hnmr_embeddings


def load_models(
    configs_path: str,
    ir_experiment: str,
    cnmr_experiment: str,
    hnmr_experiment: str,
):
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


@cache
def compute_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.RDKFingerprint(mol, maxPath=8, fpSize=2048)


@cache
def tanimoto_similarity(smiles1, smiles2):
    fp1 = compute_fingerprint(smiles1)
    fp2 = compute_fingerprint(smiles2)
    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)


@cache
def sascore(smiles):
    with contextlib.suppress(Exception):
        m = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(m)


@cache
def get_number_of_topologically_distinct_atoms(smiles: str, atomic_number: int = 1):
    """Return the number of unique `element` environments based on environmental topology.
    This corresponds to the number of peaks one could maximally observe in an NMR spectrum.
    Args:
        smiles (str): SMILES string
        atomic_number (int, optional): Atomic number. Defaults to 1.

    Returns:
        int: Number of unique environments.

    Raises:
        ValueError: If not a valid SMILES string

    Example:
        >>> get_number_of_topologically_distinct_atoms("CCO", 1)
        3

        >>> get_number_of_topologically_distinct_atoms("CCO", 6)
        2
    """

    try:
        molecule = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(molecule) if atomic_number == 1 else molecule
        # Get unique canonical atom rankings
        atom_ranks = list(Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

        # Select the unique element environments
        atom_ranks = np.array(atom_ranks)

        # Atom indices
        atom_indices = np.array([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_number])
        return len(set(atom_ranks[atom_indices]))
    except (TypeError, ValueError, AttributeError, IndexError):
        return len(smiles)


def embedding_pruning(
    smiles: str,
    spectra_ir_embedding: Tensor,
    spectra_cnmr_embedding: Tensor,
    spectra_hnmr_embedding: Tensor,
    ir_model: MolBind,
    cnmr_model: MolBind,
    hnmr_model: MolBind,
    ir_ratio: float,
    cnmr_ratio: float,
    hnmr_ratio: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    ir_embedding, cnmr_embedding, hnmr_embedding = gpu_encode_smiles(
        individual=smiles,
        ir_model=ir_model,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
    )
    # include last chunk
    chunk_range = [*list(range(0, len(smiles), 256)), len(smiles)]
    # sum the embeddings
    cosine_similarities = []
    ir_similarities = []
    cnmr_similarities = []
    hnmr_similarities = []
    cnmr_ir_similarities = []
    cnmr_hnmr_similarities = []
    ir_hnmr_similarities = []
    sum_embedding = ir_ratio * ir_embedding + cnmr_ratio * cnmr_embedding + hnmr_ratio * hnmr_embedding
    spectra_embedding_sum = spectra_ir_embedding + spectra_cnmr_embedding + spectra_hnmr_embedding
    for i, j in enumerate(chunk_range[:-1]):
        cosine_similarities.append(
            cosine_similarity(
                spectra_embedding_sum,
                sum_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        ir_similarities.append(
            cosine_similarity(
                spectra_ir_embedding,
                ir_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        cnmr_similarities.append(
            cosine_similarity(
                spectra_cnmr_embedding,
                cnmr_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        hnmr_similarities.append(
            cosine_similarity(
                spectra_hnmr_embedding,
                hnmr_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        ir_hnmr_similarities.append(
            cosine_similarity(
                spectra_ir_embedding + spectra_hnmr_embedding,
                ir_embedding[j : chunk_range[i + 1]] + hnmr_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        cnmr_ir_similarities.append(
            cosine_similarity(
                spectra_cnmr_embedding + spectra_ir_embedding,
                cnmr_embedding[j : chunk_range[i + 1]] + ir_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        cnmr_hnmr_similarities.append(
            cosine_similarity(
                spectra_cnmr_embedding + spectra_hnmr_embedding,
                cnmr_embedding[j : chunk_range[i + 1]] + hnmr_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )

    return (
        torch.cat(cosine_similarities, dim=0),
        torch.cat(ir_similarities, dim=0),
        torch.cat(cnmr_similarities, dim=0),
        torch.cat(hnmr_similarities, dim=0),
        torch.cat(cnmr_ir_similarities, dim=0),
        torch.cat(ir_hnmr_similarities, dim=0),
        torch.cat(cnmr_hnmr_similarities, dim=0),
    )


def main(
    file_with_isomers: str,
    pruned_file: str,
    index_of_smiles_to_test: int,
    dataset_path: str,
    ir_embeddings_path: str,
    cnmr_embeddings_path: str,
    hnmr_embeddings_path: str,
    ir_model: MolBind,
    cnmr_model: MolBind,
    hnmr_model: MolBind,
    folder_results: str,
    ir_ratio: float = 1,
    cnmr_ratio: float = 1,
    hnmr_ratio: float = 1,
    synthetic_access_quantile: float | None = None,
) -> None:
    index_of_smiles_to_test = int(index_of_smiles_to_test)
    ir_embeddings = pd.read_pickle(ir_embeddings_path)
    c_nmr_embeddings = pd.read_pickle(cnmr_embeddings_path)
    h_nmr_embeddings = pd.read_pickle(hnmr_embeddings_path)
    # sum up spectra embeddings
    list_of_smiles = pd.read_pickle(dataset_path).smiles.to_list()
    original_smiles = list_of_smiles[index_of_smiles_to_test]
    # print molecular formula of the original smiles
    # print the original smilesa
    logger.debug(f"Original smiles: {original_smiles}")
    ir_embeddings = aggregate_embeddings(embeddings=ir_embeddings, modalities=["smiles", "ir"], central_modality="smiles")
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

    spectra_ir_embedding = ir_embeddings["ir"][index_of_smiles_to_test].to("cuda")
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
        ir_similarities,
        cnmr_similarities,
        hnmr_similarities,
        cnmr_ir_similarities,
        ir_hnmr_similarities,
        cnmr_hnmr_similarities,
    ) = embedding_pruning(
        spectra_ir_embedding=spectra_ir_embedding,
        spectra_cnmr_embedding=spectra_cnmr_embedding,
        spectra_hnmr_embedding=spectra_hnmr_embedding,
        ir_model=ir_model,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
        smiles=isomer_df["canonical_smiles"].to_list(),
        ir_ratio=ir_ratio,
        cnmr_ratio=cnmr_ratio,
        hnmr_ratio=hnmr_ratio,
    )
    cosine_similarities = cosine_similarities.tolist()
    ir_similarities = ir_similarities.tolist()
    cnmr_similarities = cnmr_similarities.tolist()
    hnmr_similarities = hnmr_similarities.tolist()
    cnmr_ir_similarities = cnmr_ir_similarities.tolist()
    ir_hnmr_similarities = ir_hnmr_similarities.tolist()
    cnmr_hnmr_similarities = cnmr_hnmr_similarities.tolist()
    isomer_df["ir_similarity"] = ir_similarities
    isomer_df["cnmr_similarity"] = cnmr_similarities
    isomer_df["hnmr_similarity"] = hnmr_similarities
    isomer_df["similarity"] = cosine_similarities
    isomer_df["sum_of_similarities"] = isomer_df["ir_similarity"] + isomer_df["cnmr_similarity"] + isomer_df["hnmr_similarity"]
    isomer_df["cnmr_ir_similarity"] = cnmr_ir_similarities
    isomer_df["ir_hnmr_similarity"] = ir_hnmr_similarities
    isomer_df["cnmr_hnmr_similarity"] = cnmr_hnmr_similarities
    # tanimoto similarity with the original smiles
    tanimoto = partial(tanimoto_similarity, original_smiles)
    isomer_df["tanimoto"] = isomer_df["canonical_smiles"].progress_apply(lambda x: tanimoto(x))
    # save backup
    csv_path = Path(folder_results) / Path(
        str(index_of_smiles_to_test) + "_" + str(Path(pruned_file).with_suffix("")) + "_sim"
    ).with_suffix(".csv")

    isomer_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    fire.Fire(main)
