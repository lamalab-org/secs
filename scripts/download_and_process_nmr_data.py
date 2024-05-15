import pandas as pd  # noqa: I001, I002, RUF100
from rdkit import Chem


def canon_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def remove_spaces_until_1H_nmr(nmr: str) -> str:
    index = nmr.find("1HNMR")
    return nmr[: index - 1].replace(" ", "") + nmr[index - 1 :]


def read_data_from_github(path_to_save: str) -> None:
    # tgt is the target data
    url_1 = "https://raw.githubusercontent.com/pak611/NMR_LSTM/main/data/tgt-train.txt"
    # smiles is the source data
    url_2 = "https://raw.githubusercontent.com/pak611/NMR_LSTM/main/data/src-train.txt"
    # these will be used to train MolBind without any prior fine-tuning
    smiles = pd.read_csv(url_1, header=None)
    nmr = pd.read_csv(url_2, header=None)
    # place cols side by side
    data = pd.concat([smiles, nmr], axis=1)
    data.columns = ["smiles", "nmr"]
    # remove spaces from smiles column
    data["smiles"] = data["smiles"].apply(lambda x: x.replace(" ", ""))
    # remove spaces from nmr column only until 1H NMR
    data["nmr"] = data["nmr"].apply(remove_spaces_until_1H_nmr)
    data.to_csv(path_to_save, index=False)


if __name__ == "__main__":
    read_data_from_github(path_to_save="../data/nmr_data.csv")
