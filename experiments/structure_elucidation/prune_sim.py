from functools import cache

import rootutils
import torch
from hydra import compose, initialize
from rdkit import Chem, DataStructs
from torch import Tensor
from tqdm import tqdm

from molbind.data.available import ModalityConstants
from molbind.models import MolBind
from molbind.utils import rename_keys_with_prefix

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
tqdm.pandas()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def read_weights(config):
    model = MolBind(config).to("cuda")
    model.load_state_dict(
        rename_keys_with_prefix(torch.load(config.ckpt_path, map_location=torch.device("cuda"))["state_dict"]),
        strict=True,
    )
    model.eval()
    return model


def tokenize_string(
    smiles: list[str],
    modality: str,
) -> tuple[Tensor, Tensor]:
    tokenized_data = ModalityConstants[modality].tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=128,
    )
    return tokenized_data["input_ids"], tokenized_data["attention_mask"]


def encode_smiles(individual, ir_model, cnmr_model, hnmr_model):
    input_ids, attention_mask = tokenize_string(individual, "smiles")
    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    with torch.no_grad():
        ir_embedding = ir_model.encode_modality((input_ids, attention_mask), modality="smiles")
        cnmr_embedding = cnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")
        hnmr_embedding = hnmr_model.encode_modality((input_ids, attention_mask), modality="smiles")

    return ir_embedding, cnmr_embedding, hnmr_embedding


def gpu_encode_smiles(individual, ir_model, cnmr_model, hnmr_model):
    # split into chunks of: chunk_size
    chunk_size = 4096
    ir_embeddings = []
    cnmr_embeddings = []
    hnmr_embeddings = []
    chunk_range = [*list(range(0, len(individual), chunk_size)), len(individual)]
    for i, j in enumerate(tqdm(chunk_range[:-1])):
        ir_embedding, cnmr_embedding, hnmr_embedding = encode_smiles(
            individual[j : chunk_range[i + 1]],
            ir_model,
            cnmr_model,
            hnmr_model,
        )
        torch.cuda.empty_cache()
        ir_embeddings.append(ir_embedding)
        cnmr_embeddings.append(cnmr_embedding)
        hnmr_embeddings.append(hnmr_embedding)
    ir_embeddings = torch.cat(ir_embeddings, dim=0)
    cnmr_embeddings = torch.cat(cnmr_embeddings, dim=0)
    hnmr_embeddings = torch.cat(hnmr_embeddings, dim=0)
    return ir_embeddings, cnmr_embeddings, hnmr_embeddings


def load_models(
    configs_path: str,
    ir_experiment: str,
    cnmr_experiment: str,
    hnmr_experiment: str,
):
    # convert to str
    configs_path = str(configs_path)
    ir_experiment = str(ir_experiment)
    cnmr_experiment = str(cnmr_experiment)
    with initialize(version_base="1.3", config_path=configs_path):
        ir_config = compose(config_name="molbind_config", overrides=[f"experiment={ir_experiment}"])
    with initialize(version_base="1.3", config_path=configs_path):
        hnmr_config = compose(
            config_name="molbind_config",
            overrides=[f"experiment={hnmr_experiment}"],
        )
    with initialize(version_base="1.3", config_path=configs_path):
        cnmr_config = compose(
            config_name="molbind_config",
            overrides=[f"experiment={cnmr_experiment}"],
        )

    ir_model = read_weights(ir_config)
    cnmr_model = read_weights(cnmr_config)
    hnmr_model = read_weights(hnmr_config)
    return ir_model, cnmr_model, hnmr_model


@cache
def compute_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.RDKFingerprint(mol, maxPath=8, fpSize=2048)


@cache
def tanimoto_similarity(smiles1, smiles2):
    fp1 = compute_fingerprint(smiles1)
    fp2 = compute_fingerprint(smiles2)
    return DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity)


def embedding_pruning(
    smiles: list[str],
    spectra_ir_embedding: Tensor,
    spectra_cnmr_embedding: Tensor,
    spectra_hnmr_embedding: Tensor,
    ir_model: MolBind,
    cnmr_model: MolBind,
    hnmr_model: MolBind,
    ir_ratio: float = 1.0,
    cnmr_ratio: float = 1.0,
    hnmr_ratio: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    ir_embedding, cnmr_embedding, hnmr_embedding = gpu_encode_smiles(
        individual=smiles,
        ir_model=ir_model,
        cnmr_model=cnmr_model,
        hnmr_model=hnmr_model,
    )
    # include last chunk
    chunk_range = [*list(range(0, len(smiles), 8192)), len(smiles)]
    # sum the embeddings
    cosine_similarities, ir_similarities, cnmr_similarities, hnmr_similarities = [], [], [], []
    sum_embedding = ir_ratio * ir_embedding + cnmr_ratio * cnmr_embedding + hnmr_ratio * hnmr_embedding
    spectra_embedding_sum = spectra_ir_embedding + spectra_cnmr_embedding + spectra_hnmr_embedding

    for i, j in enumerate(chunk_range[:-1]):
        cosine_similarities.append(
            cosine_similarity(
                spectra_embedding_sum,
                sum_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        ir_similarities.append(
            cosine_similarity(
                spectra_ir_embedding,
                ir_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        cnmr_similarities.append(
            cosine_similarity(
                spectra_cnmr_embedding,
                cnmr_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )
        hnmr_similarities.append(
            cosine_similarity(
                spectra_hnmr_embedding,
                hnmr_embedding[j : chunk_range[i + 1]],
            )
            .detach()
            .cpu()
        )

    return (
        torch.cat(cosine_similarities, dim=0),
        torch.cat(ir_similarities, dim=0),
        torch.cat(cnmr_similarities, dim=0),
        torch.cat(hnmr_similarities, dim=0),
    )
