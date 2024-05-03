import fire  # noqa: I002
import pandas as pd
from datasets import load_dataset


def download_data(dataset_name: str, save_dir: str) -> None:
    dataset = load_dataset(dataset_name, column_names=["smiles"])

    molclr_smiles_dataset = pd.DataFrame(dataset["train"])
    molclr_smiles_dataset.to_csv(save_dir, index=False)


if __name__ == "__main__":
    fire.Fire(download_data)
