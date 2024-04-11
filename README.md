<div align="center">

# MolBind

</div>

## Installation guide

It is recommmended using `mamba` or `conda` for creating a virtual environment.

```conda
conda env create -f environment.yaml
conda activate molbind
```

## Train the model

The training script is available in `experiments/`. Define your `WANDB_PROJECT` and `WANDB_ENTITY` in the `.env` file that is read into the training script trough:

```python
load_dotenv("path/to/.env")
```