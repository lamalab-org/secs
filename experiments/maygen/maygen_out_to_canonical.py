import contextlib
import json
from pathlib import Path

import fire
import pandas as pd
from rdkit.Chem import CanonSmiles
from tqdm import tqdm

tqdm.pandas()


def read_txt_as_df(file_path):
    with Path(file_path).open("r") as f:
        return f.read().splitlines()


def canon_smiles(smi):
    with contextlib.suppress(Exception):
        return CanonSmiles(smi)
    return None


def lines_to_df(lines):
    """File is just a list of SMILES strings"""
    return pd.DataFrame({"smiles": lines})


def process_txt_input(input_file: str, output_file: str):
    lines = read_txt_as_df(input_file)
    isomer_df = lines_to_df(lines)
    isomer_df["canonical_smiles"] = isomer_df["smiles"].progress_apply(canon_smiles)
    isomer_df = isomer_df.dropna()
    isomer_df.to_csv(output_file, sep=",", index=False)


def process_pubchem_json(input_file: str, output_file: str):
    with Path(input_file).open("r") as f:
        data = json.load(f)
    smiles = [entry["canonicalsmiles"] for entry in data]
    canonical_smiles = [canon_smiles(smi) for smi in smiles]
    json_data_df = pd.DataFrame({"smiles": smiles, "canonical_smiles": canonical_smiles})
    json_data_df = json_data_df.dropna()
    json_data_df.to_csv(output_file, sep=",", index=False)


def process_pubchem_csv(input_file: str, output_file: str):
    isomer_df = pd.read_csv(input_file)

    isomer_df["canonical_smiles"] = isomer_df["canonicalsmiles"].progress_apply(canon_smiles)
    isomer_df = isomer_df.dropna()
    isomer_df.to_csv(output_file, sep=",", index=False)


def isomer_to_canonical(input_file: str, output_file: str):
    file_type = Path(input_file).suffix
    if file_type == ".txt":
        process_txt_input(input_file, output_file)
    elif file_type == ".json":
        process_pubchem_json(input_file, output_file)
    elif file_type == ".csv":
        process_pubchem_csv(input_file, output_file)


if __name__ == "__main__":
    fire.Fire(isomer_to_canonical)
