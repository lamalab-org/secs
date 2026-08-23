from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from loguru import logger
from torch import Tensor
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch_geometric.data import Batch

from secs.data.modalities import ModalityConstants
from secs.models import MolBind
from secs.utils import select_device
from secs.utils.graph import smiles_to_graph_data


def molbind_state_dict(state_dict: dict) -> dict:
    """Reduce a SECSModule checkpoint to just the MolBind weights.

    The checkpoint keys are relative to the LightningModule ("model.<...>");
    MolBind is that `model` attribute, so the prefix has to come off.
    """
    renamed = dict(state_dict)
    consume_prefix_in_state_dict_if_present(renamed, "model.")
    return renamed


def load_model(config, device: str) -> MolBind:
    """Build a MolBind from a composed Hydra config and load its checkpoint.

    Loads strictly, so a genuine architecture/config mismatch still fails
    loudly rather than silently leaving layers at their initial values.
    """
    model = MolBind(config).to(device)
    checkpoint = torch.load(config.ckpt_path, map_location=torch.device(device))
    model.load_state_dict(molbind_state_dict(checkpoint["state_dict"]), strict=True)
    model.eval()
    return model


def load_models(
    configs_path: str | Path,
    experiments: dict[str, str | None],
    device: str | None = None,
) -> dict[str, MolBind]:
    """Load one model per modality from Hydra experiment names.

    Modalities whose experiment is None, or which fail to load, are omitted
    from the result rather than mapped to None -- callers previously had to
    filter Nones out again at every use site.
    """
    device = device or select_device()
    config_dir = str(Path(configs_path).resolve())
    models: dict[str, MolBind] = {}
    for modality, experiment in experiments.items():
        if not experiment:
            continue
        try:
            with initialize_config_dir(version_base="1.3", config_dir=config_dir):
                config = compose(config_name="molbind_config", overrides=[f"experiment={experiment}"])
            models[modality] = load_model(config, device)
            logger.info(f"Loaded {modality} model from experiment: {experiment}")
        except Exception as error:
            logger.warning(f"Failed to load {modality} model ({experiment}): {error}")
    return models


class SmilesEmbedder:
    """Encodes SMILES into each modality's shared embedding space.

    One MolBind per modality. Each model is entered through its own *central*
    tower -- the MolFormer SMILES encoder for a smiles-central model, the MolCLR
    GIN over the RDKit graph for a graph-central one -- so a candidate structure
    can be compared against a target spectrum embedding whichever way the model
    was trained. Callers only ever hand over SMILES strings.
    """

    def __init__(
        self,
        models: dict[str, MolBind],
        device: str | None = None,
        context_length: int = 128,
        chunk_size: int = 8192,
    ) -> None:
        self.models = models
        self.device = device or select_device()
        self.context_length = context_length
        self.chunk_size = chunk_size
        self._tokenizer = ModalityConstants["smiles"].tokenizer
        unsupported = {
            m: model.central_modality for m, model in models.items() if model.central_modality not in ("smiles", "graph")
        }
        if unsupported:
            raise ValueError(f"Unsupported central modality for candidate encoding: {unsupported}")

    @property
    def modalities(self) -> list[str]:
        return list(self.models)

    def tokenize(self, smiles: list[str]) -> tuple[Tensor, Tensor]:
        tokens = self._tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.context_length,
        )
        return tokens["input_ids"], tokens["attention_mask"]

    def _encode_smiles_central(self, model: MolBind, smiles: list[str]) -> Tensor:
        input_ids, attention_mask = self.tokenize(smiles)
        return model.encode_modality((input_ids.to(self.device), attention_mask.to(self.device)), modality="smiles")

    def _encode_graph_central(self, model: MolBind, smiles: list[str]) -> Tensor:
        """Graph tower: RDKit graphs batched with torch_geometric.

        Candidates a generator proposes are not all parseable. Those get a zero
        embedding (cosine similarity 0) rather than breaking the batch; the
        validity penalty is what actually punishes them.
        """
        graphs = [smiles_to_graph_data(s) for s in smiles]
        valid = [i for i, g in enumerate(graphs) if g is not None]
        # Encode methane when nothing parses, purely to learn the embedding width.
        batch_graphs = [graphs[i] for i in valid] if valid else [smiles_to_graph_data("C")]
        encoded = model.encode_modality(Batch.from_data_list(batch_graphs).to(self.device), modality="graph")
        out = torch.zeros(len(smiles), encoded.shape[-1], device=encoded.device, dtype=encoded.dtype)
        if valid:
            out[valid] = encoded
        return out

    def _encode_batch(self, smiles: list[str]) -> dict[str, Tensor]:
        embeddings: dict[str, Tensor] = {}
        with torch.inference_mode():
            for modality, model in self.models.items():
                if model.central_modality == "graph":
                    embeddings[modality] = self._encode_graph_central(model, smiles)
                else:
                    embeddings[modality] = self._encode_smiles_central(model, smiles)
        return embeddings

    def encode(self, smiles: list[str]) -> dict[str, Tensor]:
        """Encode SMILES in chunks. Returns {modality: (N, D)} on the CPU.

        Results come back on the CPU so large candidate sets do not pin GPU
        memory between generations.
        """
        if not smiles or not self.models:
            return {modality: torch.empty(0) for modality in self.models}

        parts: dict[str, list[Tensor]] = {modality: [] for modality in self.models}
        for start in range(0, len(smiles), self.chunk_size):
            chunk = smiles[start : start + self.chunk_size]
            for modality, embedding in self._encode_batch(chunk).items():
                parts[modality].append(embedding.cpu())

        return {modality: torch.cat(tensors, dim=0) for modality, tensors in parts.items()}
