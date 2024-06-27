import os  # noqa: I002
from enum import Enum, StrEnum
from pathlib import Path
from typing import Dict, Optional  # noqa: UP035

import numpy as np
import pandas as pd
import torch
from hpsklearn import HyperoptEstimator, random_forest_classifier
from hyperopt import tpe
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from tqdm.auto import tqdm

from molbind.data.utils import (
    canonicalize_smiles,
    create_scaffold_split,
    smiles_to_selfies,
)
from molbind.data.utils.fingerprint_utils import compute_fragprint
from molbind.utils import select_device

tqdm.pandas()


class MoleculeNetSplit(StrEnum):
    """
    As recommended in: https://moleculenet.org/datasets
    Enum class for MoleculeNet splits.
    """

    BACE = "scaffold"
    BBBP = "scaffold"
    CLINTOX = "random"
    HIV = "scaffold"
    MUV = "random"
    SIDER = "random"
    TOX21 = "random"
    QM9 = "random"


class MoleculeNetTaskType(StrEnum):
    BACE = "classification"
    BBBP = "classification"
    CLINTOX = "classification"
    HIV = "classification"
    MUV = "classification"
    SIDER = "classification"
    TOX21 = "classification"
    QM9 = "regression"


class MoleculeNetTargetList(Enum):
    BACE = ("Class",)
    BBBP = ("p_np",)
    CLINTOX = ("CT_TOX", "FDA_APPROVED")
    HIV = ("HIV_active",)
    MUV = (
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    )

    SIDER = (
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    )
    TOX21 = (
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    )

    QM9 = (
        "mu",
        "alpha",
        "homo",
        "lumo",
        "gap",
        "r2",
        "zpve",
        "cv",
        "u0",
        "u298",
        "h298",
        "g298",
    )


class MoleculeNetTask(StrEnum):
    BACE = "bace"
    BBBP = "BBBP"
    CLINTOX = "clintox"
    HIV = "HIV"
    MUV = "muv"
    SIDER = "sider"
    TOX21 = "tox21"
    QM9 = "qm9"


class MoleculeNetURL(StrEnum):
    BACE = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.BACE}.csv"
    BBBP = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.BBBP}.csv"
    HIV = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.HIV}.csv"
    QM9 = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.QM9}.csv"


class MoleculeNetZippedURL(StrEnum):
    CLINTOX = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.CLINTOX}.csv.gz"
    MUV = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.MUV}.csv.gz"
    SIDER = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.SIDER}.csv.gz"
    TOX21 = f"https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{MoleculeNetTask.TOX21}.csv.gz"


def download_moleculenet_task(task: MoleculeNetTask) -> pd.DataFrame:
    task_name = task.upper()
    if Path(f"{task}.csv").exists():
        return pd.read_csv(f"{task}.csv")
    logger.info(f"Downloading {task_name}...")
    if hasattr(MoleculeNetZippedURL, task_name):
        os.system(f"wget {MoleculeNetZippedURL[task_name]}")
        os.system(f"gzip -d {task}.csv.gz")
    elif hasattr(MoleculeNetURL, task_name):
        os.system(f"wget {MoleculeNetURL[task_name]}")
    else:
        raise ValueError(f"Task {task} not found")
    # canonicalize smiles
    task_df = pd.read_csv(f"{task}.csv")
    if task_name == "BACE":
        task_df = task_df.rename(columns={"mol": "smiles"})
    task_df["smiles"] = task_df["smiles"].apply(canonicalize_smiles)
    task_df.to_csv(f"{task}.csv", index=False)
    return task_df


def compute_roc_auc(
    model: RandomForestClassifier, X_test: np.array, y_test: np.array
) -> Dict[str, float]:  # noqa: UP006
    y_pred = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred)


def random_forest_hyperopt(
    task_name: MoleculeNetTask,
    embedding_model: str,
    train_embeddings: np.array,
    train: pd.DataFrame,
    valid_embeddings: np.array = None,
    valid: pd.DataFrame = None,
    max_evals: int = 50,
    trial_timeout: int = 300,
    one_model_for_all: bool = False,
) -> Dict[str, RandomForestClassifier]:  # noqa: UP006
    def loss_fn_wrapper_kappa(y_true: np.array, y_pred: np.array) -> float:
        return 1 - cohen_kappa_score(y_true, y_pred)

    dict_best_models = {}

    for task_index, task in enumerate(MoleculeNetTargetList[task_name.upper()].value):
        if one_model_for_all and task_index == 0:
            # optimize the random forest classifier
            estim = HyperoptEstimator(
                classifier=random_forest_classifier(
                    name=f"{task_name}_{embedding_model}",
                    random_state=42,
                    n_jobs=-1,
                ),
                preprocessing=[],
                loss_fn=loss_fn_wrapper_kappa,
                algo=tpe.suggest,
                max_evals=max_evals,
                trial_timeout=trial_timeout,
                n_jobs=-1,
                seed=42,
            )
            y_train = train[task].dropna().to_numpy()
            estim.fit(train_embeddings, y_train)
            best_model = estim.best_model()["learner"]
            logger.info(f"Model for tasks: {best_model}")
        elif not one_model_for_all:
            estim = HyperoptEstimator(
                classifier=random_forest_classifier(
                    name=f"{task_name}_{embedding_model}",
                    n_estimators=250,  # use quarter of parameters
                    criterion="entropy",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
                preprocessing=[],
                loss_fn=loss_fn_wrapper_kappa,
                algo=tpe.suggest,
                max_evals=max_evals,
                trial_timeout=trial_timeout,
                n_jobs=-1,
                seed=42,
            )
            y_train = train[task].dropna().to_numpy()
            estim.fit(train_embeddings, y_train)
            best_model = estim.best_model()["learner"]
            logger.info(f"Best model for {task}: {estim.best_model()}")

        if valid_embeddings is not None and valid is not None:
            y_valid = valid[task].reset_index().dropna()
            import pdb

            pdb.set_trace()
            roc_auc = compute_roc_auc(
                model=best_model,
                X_test=valid_embeddings[y_valid.index],
                y_test=y_valid.to_numpy(),
            )
            dict_best_models[task] = {"best_model": best_model, "roc_auc": roc_auc}
            logger.info(f"ROC AUC for {task}: {roc_auc}")
    return dict_best_models


def aggregate_moleculenet_task_results() -> pd.DataFrame:
    tasks = [
        MoleculeNetTask.BACE,
        MoleculeNetTask.BBBP,
        MoleculeNetTask.CLINTOX,
        MoleculeNetTask.HIV,
        MoleculeNetTask.MUV,
        MoleculeNetTask.SIDER,
        MoleculeNetTask.TOX21,
    ]
    tasks = [download_moleculenet_task(task) for task in tasks]


def prep_split(
    molnet_data: pd.DataFrame, task: MoleculeNetTask, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = [0.8, 0.1, 0.1]

    # return X_train, y_train, X_test, y_test

    if MoleculeNetSplit[task] == "scaffold":
        scaffold_split_dict = create_scaffold_split(
            df=molnet_data, frac=split, entity="smiles", seed=seed
        )
        return (
            molnet_data.loc[scaffold_split_dict["train"]],
            molnet_data.loc[scaffold_split_dict["test"]],
            molnet_data.loc[scaffold_split_dict["valid"]],
        )
    else:
        # shuffle data but keep index
        molnet_data = molnet_data.sample(frac=1, random_state=seed)
        # calculate split indices
        train_end = int(len(molnet_data) * split[0])
        valid_end = train_end + int(len(molnet_data) * split[1])
        # split the data
        train = molnet_data.iloc[:train_end]
        valid = molnet_data.iloc[train_end:valid_end]
        test = molnet_data.iloc[valid_end:]
        return train, test, valid


def prep_molecule_net_dataset(data: pd.DataFrame, task: MoleculeNetTask) -> None:
    task = task.upper()
    tasks = list(MoleculeNetTargetList[task].value)
    # Load dataz
    data = data[["smiles", *tasks]]
    # canonicalize smiles
    data["smiles"] = data["smiles"].apply(canonicalize_smiles)
    data = data.dropna(subset=["smiles"])
    data["selfies"] = data["smiles"].apply(smiles_to_selfies)
    # Add fingerprint column
    data["fingerprint"] = data["smiles"].progress_apply(
        lambda smi: compute_fragprint(smi)[0]
    )

    data["graph"] = data["smiles"].apply(lambda x: x)
    # save to pkl
    # data.to_pickle(f"{task}.pkl")
    return data.dropna(subset=["smiles", "selfies"]).reset_index(drop=False)


def aggregate_embeddings(
    embeddings: list[dict[str, torch.Tensor]],
    modalities: list[str],
    central_modality: str,
) -> Dict[str, torch.Tensor]:  # noqa: UP006
    device = select_device()
    constr_dict = {modality: [] for modality in modalities}
    central_mod_embed = {}
    for embedding_dict in embeddings:
        for modality in modalities:
            if modality in embedding_dict:
                constr_dict[modality].append(embedding_dict)
    for modality, embeds in constr_dict.items():
        if modality == modalities[0]:
            central_mod_embed[central_modality] = torch.cat(
                [predict_dict[central_modality] for predict_dict in embeds], dim=0
            ).to(device)
        constr_dict[modality] = torch.cat(
            [predict_dict[modality] for predict_dict in embeds], dim=0
        ).to(device)
    constr_dict[central_modality] = central_mod_embed[central_modality]
    return constr_dict


def logistic_regression(
    task_name: MoleculeNetTask,
    train_embeddings: np.array,
    train: pd.DataFrame,
    valid_embeddings: np.array,
    valid: pd.DataFrame,
    class_weight: Optional[str] = None,
) -> None:
    task_metrics = {}
    tasks = list(MoleculeNetTargetList[task_name.upper()].value)
    # add embeddings to dataframe

    for task in tasks:
        if task not in train.columns:
            logger.warning(f"Task {task} not in train columns")
            continue
        train_task = train.copy()
        # reset index
        train_task = train_task.reset_index(drop=True)
        valid_task = valid.copy()
        # reset index
        valid_task = valid_task.reset_index(drop=True)
        y_train = train_task[task].dropna()
        train_embeddings_task = train_embeddings[y_train.index]
        # select embeddings based on index of y_train
        y_valid = valid_task[task].dropna()
        valid_embeddings_task = valid_embeddings[y_valid.index]

        # train logistic regression
        clf = LogisticRegression(
            random_state=42, max_iter=30000, class_weight=class_weight
        )
        clf.fit(train_embeddings_task, y_train.to_numpy())
        # predict
        y_pred = clf.predict(valid_embeddings_task)
        try:
            roc_auc = roc_auc_score(y_valid.to_numpy(), y_pred)
            task_metrics[task] = {"roc_auc": roc_auc}
            logger.info(f"ROC AUC for {task}: {roc_auc}")
        except ValueError:
            logger.warning(f"Multiple classes in {task}")
            continue
    return task_metrics


def linear_regression(
    task_name: MoleculeNetTask,
    train_embeddings: np.array,
    train: pd.DataFrame,
    valid_embeddings: np.array,
    valid: pd.DataFrame,
) -> None:
    task_metrics = {}
    tasks = list(MoleculeNetTargetList[task_name.upper()].value)
    # add embeddings to dataframe

    for task in tasks:
        if task not in train.columns:
            logger.warning(f"Task {task} not in train columns")
            continue
        train_task = train.copy()
        # reset index
        train_task = train_task.reset_index(drop=True)
        valid_task = valid.copy()
        # reset index
        valid_task = valid_task.reset_index(drop=True)
        y_train = train_task[task].dropna()
        train_embeddings_task = train_embeddings[y_train.index]
        # select embeddings based on index of y_train
        y_valid = valid_task[task].dropna()
        valid_embeddings_task = valid_embeddings[y_valid.index]

        # train logistic regression
        clf = LinearRegression()
        clf.fit(train_embeddings_task, y_train.to_numpy())
        # predict
        y_pred = clf.predict(valid_embeddings_task)
        # compute mae
        mae = np.mean(np.abs(y_valid.to_numpy() - y_pred))
        rmse = np.sqrt(np.mean((y_valid.to_numpy() - y_pred) ** 2))
        task_metrics[task] = {"mae": mae, "rmse": rmse}
        logger.info(f"MAE for {task}: {mae}")
    return task_metrics
