import hydra
import os  # noqa: I002
import pytorch_lightning as L

from dotenv import load_dotenv
from lightning import Trainer
from omegaconf import DictConfig

from molbind.data.utils.graph_utils import get_train_valid_loaders_from_dataset
from molbind.models.components.molclr_lightningmodule import GCNModule
from molbind.utils.instantiators import instantiate_loggers


load_dotenv(".env")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_molclr.yaml")
def main(cfg: DictConfig):
    # loss device, batch_size, temperature, use_cosine_similarity
    """
    References:
        config ref: https://github.com/yuyangw/MolCLR/blob/master/config.yaml
        data   ref: https://codeocean.com/capsule/6901415/tree/v1
        model  ref: https://github.com/yuyangw/MolCLR/blob/master/models/gcn_molclr.py
    """

    train_dataloader, val_dataloader = get_train_valid_loaders_from_dataset(
        data_path=cfg.data.data_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_size=cfg.data.valid_size,
    )

    loggers = instantiate_loggers(cfg.logger)
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=loggers,
        devices=cfg.trainer.devices,
    )
    # set-up model
    trainer.fit(
        model=GCNModule(cfg),
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
