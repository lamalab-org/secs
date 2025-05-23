from functools import cache

import torch
from hydra import compose, initialize
from loguru import logger
from rdkit import Chem, DataStructs
from torch import Tensor
from tqdm import tqdm

from molbind.data.available import ModalityConstants
from molbind.models import MolBind
from molbind.utils import rename_keys_with_prefix

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tqdm.pandas()


# --- Model Loading ---
def read_weights(config) -> MolBind:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MolBind(config).to(device)
    model.load_state_dict(
        rename_keys_with_prefix(torch.load(config.ckpt_path, map_location=torch.device(device))["state_dict"]),
        strict=True,
    )
    model.eval()
    return model


def load_models_dict(configs_path: str, experiments_dict: dict[str, str | None]) -> dict[str, MolBind | None]:
    models = {}
    for modality, experiment in experiments_dict.items():
        if experiment:
            try:
                with initialize(version_base="1.3", config_path=str(configs_path)):
                    config = compose(config_name="molbind_config", overrides=[f"experiment={experiment}"])
                models[modality] = read_weights(config)
                logger.info(f"Loaded {modality} model from experiment: {experiment}")
            except Exception as e:
                logger.warning(f"Failed to load {modality} model ({experiment}): {e}")
                models[modality] = None
        else:
            models[modality] = None  # Explicitly None if no experiment string
    return models


# --- SMILES Tokenization and Encoding ---
def tokenize_string(smiles: list[str] | str, modality_token_type: str = "smiles") -> tuple[Tensor, Tensor]:
    if isinstance(smiles, str):
        smiles = [smiles]
    # Ensure ModalityConstants uses the correct tokenizer for "smiles" type
    # This might need adjustment if ModalityConstants is not structured as expected
    try:
        tokenizer = ModalityConstants[modality_token_type].tokenizer
    except KeyError:
        logger.error(
            f"Tokenizer for modality type '{modality_token_type}' not found in ModalityConstants. Defaulting to SMILES if possible."
        )
        tokenizer = ModalityConstants["smiles"].tokenizer  # Fallback, might not be ideal

    tokens = tokenizer(smiles, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    return tokens["input_ids"], tokens["attention_mask"]


def encode_smiles_variable(individuals: list[str] | str, models_dict: dict[str, MolBind | None]) -> dict[str, Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids, attention_mask = tokenize_string(individuals, "smiles")
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    embeddings = {}
    with torch.inference_mode():
        for mod, model in models_dict.items():
            if model:
                try:
                    embeddings[mod] = model.encode_modality((input_ids, attention_mask), modality="smiles")
                except Exception as e_enc:
                    logger.error(f"Error encoding SMILES with model for modality {mod}: {e_enc}")
                    embeddings[mod] = torch.empty(0, device=device)  # Return empty on error
    return embeddings


def gpu_encode_smiles_variable(
    individuals: list[str], models_dict: dict[str, MolBind | None], chunk_size: int = 8192
) -> dict[str, Tensor]:
    active_models = {m: model for m, model in models_dict.items() if model}
    # Determine a default device, even if no active models (for returning empty tensors)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    if active_models:
        default_device = next(iter(active_models.values()))

    if not active_models or not individuals:
        return {m: torch.empty(0, device=default_device) for m in models_dict}

    all_parts = {mod: [] for mod in active_models}
    for i in tqdm(range(0, len(individuals), chunk_size), desc="Encoding SMILES (Batch)", leave=False):
        batch = individuals[i : i + chunk_size]
        if not batch:
            continue
        chunk_embs = encode_smiles_variable(batch, active_models)
        for mod, emb in chunk_embs.items():
            if emb.nelement() > 0:  # Only append non-empty embeddings
                all_parts[mod].append(emb.cpu())

    final_embeddings = {}
    for mod in models_dict:  # Iterate over original keys
        if all_parts.get(mod):
            final_embeddings[mod] = torch.cat(all_parts[mod], dim=0)
        else:
            final_embeddings[mod] = torch.empty(0, device=default_device)
    return final_embeddings


# --- Fingerprint and Similarity ---
@cache
def compute_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.RDKFingerprint(mol, maxPath=8, fpSize=2048) if mol else None


@cache
def tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    if not smiles1 or not smiles2:
        return 0.0
    fp1, fp2 = compute_fingerprint(smiles1), compute_fingerprint(smiles2)
    return DataStructs.FingerprintSimilarity(fp1, fp2) if fp1 and fp2 else 0.0


# --- Embedding Pruning (Core Comparison Logic) ---
def embedding_pruning_variable(
    smiles_to_score: list[str],
    target_1D_spectral_embeddings: dict[str, Tensor],  # Key: mod, Val: 1D Tensor (D,) for target
    models_for_scoring: dict[str, MolBind | None],  # Models to encode candidate SMILES
    modality_ratios: dict[str, float] | None = None,
    chunk_size: int = 8192,
) -> tuple[Tensor | None, dict[str, Tensor | None]]:
    if not smiles_to_score:
        return None, {}

    scoreable_modalities = [
        mod
        for mod, target_emb in target_1D_spectral_embeddings.items()
        if models_for_scoring.get(mod)
        and target_emb is not None
        and target_emb.ndim == 1
        and target_emb.nelement() > 0  # Target must be 1D and non-empty
    ]

    num_smiles = len(smiles_to_score)
    # Initialize return structures with correct device (CPU for scores)
    all_individual_scores_cpu = {
        mod: torch.zeros(num_smiles, device="cpu") if num_smiles > 0 else torch.empty(0, device="cpu")
        for mod in models_for_scoring
    }
    combined_scores_cpu = torch.zeros(num_smiles, device="cpu") if num_smiles > 0 else None

    if not scoreable_modalities:
        logger.warning("Pruning: No scoreable modalities (need 1D target & model). Returning zeros.")
        return combined_scores_cpu, all_individual_scores_cpu

    ratios = modality_ratios or dict.fromkeys(scoreable_modalities, 1.0)

    candidate_smiles_embs_dict_gpu = gpu_encode_smiles_variable(
        smiles_to_score, {mod: models_for_scoring[mod] for mod in scoreable_modalities}, chunk_size
    )

    cos_sim = torch.nn.CosineSimilarity(dim=1)
    combined_cand_parts_gpu = []

    for mod in scoreable_modalities:
        cand_embs_gpu = candidate_smiles_embs_dict_gpu.get(mod)
        # Ensure target_1D_emb_gpu is on the same device as cand_embs_gpu for cosine_similarity
        target_emb_device = cand_embs_gpu.device if cand_embs_gpu is not None else "cpu"
        target_1D_emb_gpu = target_1D_spectral_embeddings[mod].to(target_emb_device)

        if (
            cand_embs_gpu is None
            or cand_embs_gpu.nelement() == 0
            or cand_embs_gpu.shape[0] != num_smiles
            or cand_embs_gpu.shape[1] != target_1D_emb_gpu.shape[0]
        ):
            logger.warning(
                f"Pruning: Skipping {mod} due to missing/mismatched candidate embeddings or target dim. Cand shape: {cand_embs_gpu.shape if cand_embs_gpu is not None else 'None'}, Target shape: {target_1D_emb_gpu.shape}"
            )
            # all_individual_scores_cpu[mod] is already zeros
            continue

        scores = cos_sim(target_1D_emb_gpu.unsqueeze(0), cand_embs_gpu)
        all_individual_scores_cpu[mod] = scores.cpu()
        combined_cand_parts_gpu.append(ratios.get(mod, 1.0) * cand_embs_gpu)

    if combined_cand_parts_gpu:
        summed_cand_embs_gpu = torch.sum(torch.stack(combined_cand_parts_gpu, dim=0), dim=0)

        summed_target_parts_gpu = []
        # Check if modality was actually used for candidate embeddings before adding its target part
        valid_modalities_for_sum = [
            m
            for m in scoreable_modalities
            if m in candidate_smiles_embs_dict_gpu
            and candidate_smiles_embs_dict_gpu[m] is not None
            and candidate_smiles_embs_dict_gpu[m].nelement() > 0
        ]

        for mod in valid_modalities_for_sum:
            summed_target_parts_gpu.append(
                ratios.get(mod, 1.0) * target_1D_spectral_embeddings[mod].to(summed_cand_embs_gpu.device)
            )

        if summed_target_parts_gpu:
            summed_target_emb_gpu = torch.sum(torch.stack(summed_target_parts_gpu, dim=0), dim=0)
            if summed_cand_embs_gpu.shape[1] == summed_target_emb_gpu.shape[0]:
                combined_scores_cpu = cos_sim(summed_target_emb_gpu.unsqueeze(0), summed_cand_embs_gpu).cpu()
            else:
                logger.error(
                    f"Pruning: Dim mismatch for combined score sum. Target sum: {summed_target_emb_gpu.shape}, Cand sum: {summed_cand_embs_gpu.shape}"
                )
        else:
            logger.warning("Pruning: No valid target parts for combined score sum.")
    else:
        logger.warning("Pruning: No valid candidate parts for combined score sum.")

    return combined_scores_cpu, all_individual_scores_cpu
