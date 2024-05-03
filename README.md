<div align="center">

# MolBind

</div>

## Installation guide

It is recommmended using `mamba` or `conda` for creating a virtual environment.

```conda
conda env create -f environment.yaml
conda activate molbind
```

## Data availability

- FP-VAE encoder $\rightarrow$ `molbind/data/fingerprint.pkl`
- MolCLR $\rightarrow$ `molbind/scripts/download_moclr_data.py`. The script can be ran with the following command:

```bash
python download_molclr_data.py --save_dir "../data/pretrain_example.csv" --dataset_name "AdrianM0/molbind"
```

- MolBind $\rightarrow$ `molbind/data/fingerprint.pkl` (same as for the FP-VAE encoder, but it is using more columns)

## Environment file

Your environment file should look like this:

```
WANDB_PROJECT="<your-wandb-project-name>"
WANDB_ENTITY="<your-wandb-account-name>"
TOKENIZERS_PARALLELISM=False
```

After you have defined your system variables in `.env`, it is read into the script as following:

```python
load_dotenv("path/to/.env")
```

## Train models

In `experiments/` you will find several training scripts. The global `train.py` script will refer exclusively to the multimodal model, while all the other scripts refer to specific encoders.

- `fp_vae_train.py` $\rightarrow$ Train a variational autoencoder for the Morgan fingerprints
- `molclr_train.py` $\rightarrow$ Train the MolCLR model to be used as a graph encoder in `MolBind`


## Config tree

In the config tree you will find all the necessary configs for both individual encoders and the full model. Here is the config structure. Naming will be modified (WIP) for clarity. In `configs/models` and `configs/data` you will find specific configs for the different models mentioned above. For example, `config/models/fp_vae.yaml` refers to the individual VAE for fingerprints.

```
configs
├── __init__.py
├── callbacks
│   ├── default.yaml
│   ├── early_stopping.yaml
│   ├── model_checkpoint.yaml
│   ├── model_summary.yaml
│   ├── none.yaml
│   └── rich_progress_bar.yaml
├── data
│   ├── fp_vae.yaml
│   ├── molbind.yaml
│   ├── molbind_with_fps.yaml
│   └── smi_sf_fps_graph.yaml
├── debug
│   ├── default.yaml
│   ├── fdr.yaml
│   ├── limit.yaml
│   ├── overfit.yaml
│   └── profiler.yaml
├── eval.yaml
├── experiment
│   └── example.yaml
├── extras
│   └── default.yaml
├── hparams_search
│   └── mnist_optuna.yaml
├── hydra
│   └── default.yaml
├── local
├── logger
│   ├── aim.yaml
│   ├── comet.yaml
│   ├── csv.yaml
│   ├── many_loggers.yaml
│   ├── mlflow.yaml
│   ├── neptune.yaml
│   ├── tensorboard.yaml
│   └── wandb.yaml
├── model
│   ├── fp_vae.yaml
│   ├── molbind.yaml
│   ├── molbind_with_fps.yaml
│   └── smi_sf_fps_graph.yaml
├── paths
│   └── default.yaml
├── train.yaml
├── train_4mods.yaml
├── train_fp-vae.yaml
└── trainer
    ├── ddp.yaml
    ├── default.yaml
    ├── gpu.yaml
    └── mps.yaml
```