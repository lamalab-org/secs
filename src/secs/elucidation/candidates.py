from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from secs.utils.elucidation import gen_close_molformulas_from_seed


@runtime_checkable
class CandidateSource(Protocol):
    """Proposes starting molecules for a search."""

    def propose(self, target_embedding: Tensor, formula: str, n_candidates: int = 2048) -> list[str]: ...


class StaticCandidateSource:
    """Returns a fixed list. Useful for tests and for replaying a known population."""

    def __init__(self, smiles: list[str]) -> None:
        self.smiles = smiles

    def propose(self, target_embedding: Tensor, formula: str, n_candidates: int = 2048) -> list[str]:  # noqa: ARG002
        return self.smiles[:n_candidates]


class FaissCandidateSource:
    """Nearest neighbours in embedding space, restricted to plausible formulas.

    The neighbour search establishes the *ranking* (how well each database
    molecule matches the target spectrum) and the formula filter then removes
    implausible ones while preserving that ranking. Doing it the other way
    round -- filtering the whole database by formula and ignoring the
    neighbour order -- discards the spectral evidence entirely.
    """

    def __init__(
        self,
        index,
        smiles: np.ndarray,
        formulas: np.ndarray,
        n_neighbours: int = 100_000,
    ) -> None:
        if len(smiles) != len(formulas):
            raise ValueError(f"smiles ({len(smiles)}) and formulas ({len(formulas)}) must be the same length.")
        self.index = index
        self.smiles = np.asarray(smiles)
        self.formulas = np.asarray(formulas)
        self.n_neighbours = n_neighbours

    @classmethod
    def from_files(
        cls,
        index_path: str | Path,
        parquet_path: str | Path,
        n_neighbours: int = 100_000,
        allow_partial: bool = False,
    ):
        """Load a FAISS index plus the parquet holding its SMILES and formulas.

        Row *i* of the parquet must correspond to vector *i* in the index. Set
        `allow_partial` for an index still being built: it covers the first
        `ntotal` rows, so the parquet is truncated to match. Anything beyond
        that prefix is simply unreachable, not misaligned.
        """
        import faiss  # noqa: PLC0415  (heavy optional import)
        import polars as pl  # noqa: PLC0415

        table = pl.read_parquet(parquet_path, columns=["smiles", "molecular_formula"])
        index = faiss.read_index(str(index_path))
        if index.ntotal != table.height:
            if not (allow_partial and index.ntotal < table.height):
                raise ValueError(
                    f"Index has {index.ntotal} vectors but {parquet_path} has {table.height} rows; they must correspond."
                )
            logger.warning(f"Partial index: using the first {index.ntotal:,} of {table.height:,} rows.")
            table = table.head(index.ntotal)
        return cls(
            index=index,
            smiles=table["smiles"].to_numpy(),
            formulas=table["molecular_formula"].to_numpy(),
            n_neighbours=n_neighbours,
        )

    def _search(self, target_embedding: Tensor) -> np.ndarray:
        query = target_embedding.detach().cpu().reshape(1, -1).to(torch.float32).numpy()
        _distances, neighbours = self.index.search(query, self.n_neighbours)
        found = neighbours[0]
        return found[found >= 0]  # FAISS pads short results with -1

    def propose(self, target_embedding: Tensor, formula: str, n_candidates: int = 2048) -> list[str]:
        ranked = self._search(target_embedding)
        if ranked.size == 0:
            logger.warning("Candidate search returned no neighbours.")
            return []

        allowed = gen_close_molformulas_from_seed(formula)
        # np.isin over the neighbour slice only -- not over the whole database.
        keep = ranked[np.isin(self.formulas[ranked], allowed)]

        if keep.size == 0:
            logger.warning(f"No neighbour matched a plausible formula for {formula}; falling back to nearest neighbours.")
            keep = ranked

        return self.smiles[keep][:n_candidates].tolist()


class HttpCandidateSource:
    """Calls a candidate-retrieval service that wraps the index."""

    def __init__(self, url: str, timeout: float = 300.0) -> None:
        self.url = url
        self.timeout = timeout

    def propose(self, target_embedding: Tensor, formula: str, n_candidates: int = 2048) -> list[str]:
        import requests  # noqa: PLC0415

        embedding = torch.nn.functional.normalize(target_embedding.detach().cpu().flatten(), p=2, dim=0)
        response = requests.post(
            self.url,
            json={
                "mf": formula,
                "spectrum_embedding": embedding.tolist(),
                "n_candidates": n_candidates,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["smiles"][:n_candidates]
