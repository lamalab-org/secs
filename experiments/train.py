import hydra
import pytorch_lightning as L
import polars as pl
from molbind.data.dataloaders import load_combined_loader
from molbind.models.lightning_module import MolBindModule
from omegaconf import DictConfig
import torch
import rootutils
from hydra.utils import instantiate

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def train_molbind(config: DictConfig):
    config = instantiate(config)
    wandb_logger = L.loggers.WandbLogger(**config.logger.wandb)

    device_count = torch.cuda.device_count()
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="cuda",
        log_every_n_steps=10,
        logger=wandb_logger,
        devices=device_count if device_count > 1 else "auto",
        strategy="ddp" if device_count > 1 else "auto",
    )

    train_modality_data = {}
    valid_modality_data = {}

    # Example SMILES - SELFIES modality pair:
    data = pl.read_csv(config.data.dataset_path)
    shuffled_data = data.sample(
        fraction=config.data.fraction_data, shuffle=True, seed=config.data.seed
    )
    dataset_length = len(shuffled_data)
    valid_shuffled_data = shuffled_data.tail(
        int(config.data.valid_frac * dataset_length)
    )
    train_shuffled_data = shuffled_data.head(
        int(config.data.train_frac * dataset_length)
    )

    columns = shuffled_data.columns
    # extract non-central modalities
    non_central_modalities = config.data.modalities
    central_modality = config.data.central_modality

    for column in columns:
        if column in non_central_modalities:
            # drop nan for specific pair
            train_modality_pair = train_shuffled_data[
                [central_modality, column]
            ].drop_nulls()
            valid_modality_pair = valid_shuffled_data[
                [central_modality, column]
            ].drop_nulls()

            train_modality_data[column] = [
                train_modality_pair[central_modality].to_list(),
                train_modality_pair[column].to_list(),
            ]
            valid_modality_data[column] = [
                valid_modality_pair[central_modality].to_list(),
                valid_modality_pair[column].to_list(),
            ]

    combined_loader = load_combined_loader(
        data_modalities=train_modality_data,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=1,
    )

    valid_dataloader = load_combined_loader(
        data_modalities=valid_modality_data,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=1,
    )

    trainer.fit(
        MolBindModule(config),
        train_dataloaders=combined_loader,
        val_dataloaders=valid_dataloader,
    )


@hydra.main(config_path="../configs", config_name="train.yaml")
def main(config: DictConfig):
    # train_molbind(config)
    import pdb; pdb.set_trace()
    config = instantiate(config)


if __name__ == "__main__":
    main()
