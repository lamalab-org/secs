import datetime
import pickle as pkl
from pathlib import Path

import hydra
import pandas as pd
import polars as pl
import pytorch_lightning as L
import rootutils
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from molbind.data.datamodule import MolBindDataModule
from molbind.models.lightning_module import MolBindModule

load_dotenv()
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# set an unique identifier for the retrieval run
RETRIEVAL_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M")


def embed(config: DictConfig):
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        devices=1,
        strategy="auto",
    )

    # extract format of dataset file
    data_format = Path(config.data.dataset_path).suffix
    handlers = {
        ".csv": pd.read_csv,
        ".pickle": pd.read_pickle,
        ".pkl": pd.read_pickle,
        ".parquet": pl.read_parquet,
    }

    try:
        data = handlers[data_format](config.data.dataset_path).to_pandas()
    except KeyError:
        logger.error(f"Format {data_format} not supported")

    # Split the data into validation and training datasets
    dataloader = MolBindDataModule(
        data={
            "dataloader_arguments": {
                "batch_size": config.data.batch_size,
                "num_workers": config.data.num_workers,
            },
        },
    ).embed_dataloader(data.tokens.to_list())
    predictions = trainer.predict(model=MolBindModule(config), dataloaders=[dataloader])
    with open(f"/data/mirzaa/pubchem_embeddings_{RETRIEVAL_TIME}.pkl", "wb") as f:  # noqa: PTH123
        pkl.dump(predictions, f, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info(f"Saved embeddings to /data/mirzaa/pubchem_embeddings_{RETRIEVAL_TIME}.pkl")


@hydra.main(version_base="1.3", config_path="../configs", config_name="molbind_config.yaml")
def main(config: DictConfig):
    embed(config)


if __name__ == "__main__":
    main()
