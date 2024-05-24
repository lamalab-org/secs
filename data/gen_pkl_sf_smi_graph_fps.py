import fire  # noqa: I002
import pandas as pd
import selfies as sf

from molbind.data.utils.fingerprint_utils import get_morgan_fingerprint_from_smiles


def smiles_to_selfies(smiles: str) -> str:
    """
    Convert a SELFIES string to a SMILES string.

    Args:
        selfies (str): SELFIES string

    Returns:
        str: SMILES string
    """
    try:
        return sf.encoder(smiles)
    except Exception:
        return None

def add_fingerprint_column_to_dataframe(
    csv_data_path: str, radius: int = 4, nbits: int = 2048
) -> None:
    """
    Add fingerprint column to a pandas dataframe. The dataframe is saved as a
    pickle file with the same name as the csv_data_path (str) input variable,
    but with a changed extension.

    Args:
        csv_data_path (str): Path to csv file with data
        radius (int, optional): radius of the fingerprint. Defaults to 2.
        nbits (int, optional): number of bits in the fingerprint. Defaults to 2048.
    Returns:
        None
    """

    # Load data
    data = pd.read_csv(csv_data_path)
    data = data[["smiles"]]
    data["selfies"] = data["smiles"].apply(lambda x: smiles_to_selfies(x))
    print("Finished SELFIES conversion")
    # Add fingerprint column
    data["fingerprint"] = data["smiles"].apply(
        lambda smi: get_morgan_fingerprint_from_smiles(smi, radius, nbits)
    )


    data["graph"] = data["smiles"].apply(lambda x: x)
    data = data.dropna()
    # Save data to pkl
    data.to_pickle(csv_data_path.replace(".csv", ".pkl"))


if __name__ == "__main__":
    fire.Fire(add_fingerprint_column_to_dataframe)
