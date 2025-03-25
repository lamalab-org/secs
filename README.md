<div align="center">

# MoleculeBind

</div>

## :scroll: Installation guide

It is recommmended to use `uv` for creating a virtual environment. The instructions to install `uv` can be found on [uv's homepage](https://docs.astral.sh/uv/getting-started/installation/).

```conda
uv venv --python 3.11 molbind
source molbind/bin/activate
uv pip install -e .
uv pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
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

The experiment configs can be found at config. For example, to run the `train.py`

```python
python train.py 'experiment="train/ir_simulated_large_dataset"'
```

The training scripts outputs the checkpoints at `experiments/checkpoints/<run-code-name>/<checkpoint-file-name>.ckpt`
To find all three checkpoints used in this work, please access the supplementary information on [Zenodo](https://zenodo.org/records/14634449).

To run the metrics on these experiments:

```python
python retrieval.py 'experiment="metrics/ir_simulated_large_dataset"'
```

## ⚙️ System requirements

For the training script 4 NVIDIA A100-40GB GPUs have been used. For the retrieval script 1 NVIDIA A100-40GB GPUs has been used.


## 💰 Funding

This work was funded by the Carl-Zeiss Foundation. In addition, this work was partly funded by the SOL-AI project funded as part of the Helmholtz Foundation Model Initiative of the Helmholtz Association. Moreover, this work was supported by Helmholtz AI computing resources (HAICORE) of the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI.

## Citation

```bib
@article{mirza2024elucidating,
  title={Elucidating structures from spectra using multimodal embeddings and discrete optimization},
  author={Mirza, Adrian and Jablonka, Kevin Maik},
  year={2024}
}
@inproceedings{mirza2024bridging,
  title={Bridging chemical modalities by aligning embeddings},
  author={Mirza, Adrian and Starke, Sebastian and Merdivan, Erinc and Jablonka, Kevin Maik},
  booktitle={AI for Accelerated Materials Design-Vienna 2024},
  year={2024}
}
```
