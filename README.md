<div align="center">

# MolBind

</div>

## Installation guide

It is recommmended using `mamba` or `conda` for creating a virtual environment. For inference/embeddings the installation guide is given below:

```conda
conda env create -n molbind python=3.12
pip install -e .[inference]
```


If you want to (re)train the models, your system needs to have `CUDA` dependencies, please use the `environment.yaml` file for the installation.
Experimental IR spectra have been mined from the `NIST` database. The code for mining is available at [chemistry-spectra-datasets](https://github.com/lamalab-org/chemistry-spectra-dataset/tree/main).

```conda
conda env create -f environment.yaml
conda activate molbind
```

## Data availability

The simulated spectra data have been compiled from [IBM's Multimodal Spectroscopic Dataset](https://zenodo.org/records/11611178).


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

The experiment configs can be found at config
For example, to run the `train.py`

```python
python train.py 'experiment="train/ir_simulated"'
```

To run the metrics on these experiments:

```python
python retrieval.py 'experiment="metrics/ir_simulated"'
```
