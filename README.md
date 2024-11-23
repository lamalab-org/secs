<div align="center">

# MolBind

</div>

## :scroll: Installation guide

It is recommmended using `mamba` or `conda` for creating a virtual environment. For inference/embeddings the installation guide is given below:

```conda
conda create -n molbind python=3.12
pip install -e .[inference]
```

If you want to (re)train the models, your system needs to have `CUDA` dependencies, please use the `environment.yaml` file for the installation.

```conda
conda env create -f environment.yaml
conda activate molbind
```

## :file_folder: Data availability

The simulated spectra data have been compiled from [IBM's Multimodal Spectroscopic Dataset](https://zenodo.org/records/11611178).

(WIP :building_construction:) Run `molbind-get-datasets` from the command line to download the data.


## :clipboard: Environment file

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

## :chart_with_downwards_trend: Train models

The experiment configs can be found at config
For example, to run the `train.py`

```python
python train.py 'experiment="train/ir_simulated"'
```

To run the metrics on these experiments:

```python
python retrieval.py 'experiment="metrics/ir_simulated"'
```

## 💰 Funding

This work was funded by the Carl-Zeiss Foundation. In addition, this work was partly funded by the SOL-AI project funded as part of the Helmholtz Foundation Model Initiative of the Helmholtz Association. Moreover, this work was supported by Helmholtz AI computing resources (HAICORE) of the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI.
