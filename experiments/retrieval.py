import datetime
import pickle as pkl
from pathlib import Path
from pprint import pformat

import hydra
import pandas as pd
import polars as pl
import pytorch_lightning as L
import rootutils
import selfies as sf
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from molbind.data.analysis import aggregate_embeddings
from molbind.data.datamodule import MolBindDataModule
from molbind.data.molbind_dataset import MolBindDataset
from molbind.data.utils.file_utils import csv_load_function, pickle_load_function
from molbind.metrics.retrieval import full_database_retrieval
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
        ".csv": csv_load_function,
        ".pickle": pickle_load_function,
        ".pkl": pickle_load_function,
        ".parquet": pl.read_parquet,
    }

    try:
        data = handlers[data_format](config.data.dataset_path)
    except KeyError:
        logger.error(f"Format {data_format} not supported")
    data = data.with_columns(pl.col("smiles").alias("graph"))
    data = data.to_pandas()
    data["selfies"] = data["smiles"].apply(sf.encoder)
    data = pl.DataFrame(data)
    # data = data.with_columns(pl.col("smiles").alias("graph"))
    # data = data.drop_nulls(subset=["structure"])
    # data = data.with_columns(
    #     pl.col("smiles")
    #     .map_elements(compute_fragprint, return_dtype=pl.List[float])
    #     .alias("fingerprint")
    # )
    # # drop duplicates in central modality column
    # data = data.unique(subset=[config.data.central_modality])
    shuffled_data = data.sample(
        fraction=config.data.fraction_data, shuffle=True, seed=config.data.seed
    )
    dataset_length = len(shuffled_data)
    valid_shuffled_data = shuffled_data.tail(
        int(config.data.valid_frac * dataset_length)
    )
    valid_shuffled_data = valid_shuffled_data.unique(
        subset=[config.data.central_modality]
    )

    valid_datasets = (
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=valid_shuffled_data,
            context_length=config.data.context_length,
        ).build_datasets_for_modalities(),
    )

    valid_shuffled_data.to_pandas().to_pickle(f"valid_dataset_{RETRIEVAL_TIME}.pkl")
    datamodule = MolBindDataModule(
        data={
            "predict": valid_datasets,
            "dataloader_arguments": {
                "batch_size": config.data.batch_size,
                "num_workers": config.data.num_workers,
            },
        },
    )

    predictions = trainer.predict(
        model=MolBindModule(config),
        datamodule=datamodule,
    )
    with open(f"{config.embeddings_path}_{RETRIEVAL_TIME}.pkl", "wb") as f:  # noqa: PTH123
        pkl.dump(predictions, f, protocol=pkl.HIGHEST_PROTOCOL)

    aggregated_embeddings = aggregate_embeddings(
        embeddings=predictions,
        modalities=config.data.modalities,
        central_modality=config.data.central_modality,
    )

    # concatenate predictions outside of this script and save predictions
    with open(f"{config.embeddings_path}.pkl", "wb") as f:  # noqa: PTH123
        pkl.dump(aggregated_embeddings, f, protocol=pkl.HIGHEST_PROTOCOL)

    logger.info(f"Saved embeddings to {config.embeddings_path}.pkl")
    retrieval_metrics = full_database_retrieval(
        indices=valid_shuffled_data.to_pandas(),
        other_modalities=config.data.modalities,
        central_modality=config.data.central_modality,
        embeddings=aggregated_embeddings,
        top_k=config.top_k,
    )
    retrieval_metrics = pd.DataFrame(retrieval_metrics)
    retrieval_metrics.to_csv(f"{config.run_id}_retrieval_metrics.csv")
    logger.info(f"Database size: {len(valid_shuffled_data)}")
    logger.info(f"Database level retrieval metrics: \n {pformat(retrieval_metrics)}")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="molbind_config.yaml"
)
def main(config: DictConfig):
    embed(config)


if __name__ == "__main__":
    main()
