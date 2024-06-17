import pickle as pkl  # noqa: I002

import hydra
import numpy as np
import polars as pl
import pytorch_lightning as L
import rootutils
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from molbind.data.analysis.moleculenet import (
    MoleculeNetTask,
    MoleculeNetTaskType,
    aggregate_embeddings,
    download_moleculenet_task,
    linear_regression,
    logistic_regression,
    prep_molecule_net_dataset,
    prep_split,
)
from molbind.data.datamodule import MolBindDataModule
from molbind.data.molbind_dataset import MolBindDataset
from molbind.models.lightning_module import MolBindModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

load_dotenv()


def embed_dataset_and_compute_metrics(config: DictConfig):
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        devices=1,
    )

    data = download_moleculenet_task(MoleculeNetTask[config.task_name])
    data_to_embed = prep_molecule_net_dataset(data, MoleculeNetTask[config.task_name])
    # convert dataframe_to_polars
    data_to_embed_pl = pl.from_pandas(data_to_embed)
    valid_datasets = (
        MolBindDataset(
            central_modality=config.data.central_modality,
            other_modalities=config.data.modalities,
            data=data_to_embed_pl,
            context_length=config.data.context_length,
        ).build_datasets_for_modalities(),
    )

    if len(data_to_embed) < config.data.batch_size:
        config.data.batch_size = len(data_to_embed)
    else:
        config.data.batch_size = config.data.batch_size
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

    aggregated_embeddings = aggregate_embeddings(
        embeddings=predictions,
        smiles=data_to_embed["smiles"].tolist(),
        modalities=config.data.modalities,
        central_modality=config.data.central_modality,
    )

    embed_central_modality = (
        aggregated_embeddings[config.data.central_modality].detach().cpu().numpy()
    )
    # save all embeddings to file
    with open(config.store_embeddings_directory + ".pkl", "wb") as f:
        pkl.dump(aggregated_embeddings, f)

    train, test, valid = prep_split(data_to_embed, config.task_name, seed=42)
    # best_rf_models = random_forest_hyperopt(
    #     task_name=config.task_name,
    #     embedding_model=config.run_id,
    #     train_embeddings=embed_central_modality[train.index],
    #     train=train,
    #     valid_embeddings=embed_central_modality[valid.index],
    #     valid=valid,
    #     trial_timeout=500,
    #     max_evals=5,
    #     one_model_for_all=False,
    # )
    # task_average = np.mean([best_rf_models[subtask]["roc_auc"] for subtask in best_rf_models])
    # # save to file
    # logger.info(f"Task Average: {task_average}")

    if MoleculeNetTaskType[config.task_name] == "regression":
        task_metrics = linear_regression(
            task_name=config.task_name,
            train_embeddings=embed_central_modality[train.index],
            train=train,
            valid_embeddings=embed_central_modality[valid.index],
            valid=valid,
        )
        task_average = np.mean([task_metrics[subtask]["roc_auc"] for subtask in task_metrics])
    else:
        task_metrics = logistic_regression(
            task_name=config.task_name,
            train_embeddings=embed_central_modality[train.index],
            train=train,
            valid_embeddings=embed_central_modality[valid.index],
            valid=valid,
            class_weight="balanced",
        )

        task_average = np.mean([task_metrics[subtask]["mae"] for subtask in task_metrics])
        # save to file
    logger.info(f"Task Average: {task_average}")


@hydra.main(version_base="1.3", config_path="../configs")
def main(config: DictConfig) -> None:
    embed_dataset_and_compute_metrics(config)


if __name__ == "__main__":
    main()
