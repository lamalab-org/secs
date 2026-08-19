import numpy as np
from loguru import logger


class HttpShiftSimulator:
    """Calls a shift-prediction service that runs in its own environment.

    Predictors like CASCADE-2.0 pin TensorFlow and Python 3.10, which cannot
    be co-installed with this package. Running one behind HTTP -- as the
    candidate-retrieval and forward-synthesis services already are -- keeps
    that dependency set isolated.

    The service is expected to accept ``{"smiles": [...]}`` and return
    ``{"shifts": [[...], null, ...]}``, one entry per input in the same order,
    with null where prediction failed.
    """

    def __init__(
        self,
        url: str,
        modality: str = "c_nmr",
        timeout: float = 300.0,
        batch_size: int = 512,
    ) -> None:
        self.url = url
        self.modality = modality
        self.timeout = timeout
        self.batch_size = batch_size

    def _post(self, batch: list[str]) -> list[np.ndarray | None]:
        import requests  # noqa: PLC0415

        response = requests.post(self.url, json={"smiles": batch}, timeout=self.timeout)
        response.raise_for_status()
        shifts = response.json()["shifts"]
        if len(shifts) != len(batch):
            raise ValueError(f"Simulator returned {len(shifts)} predictions for {len(batch)} molecules.")
        return [None if s is None else np.asarray(s, dtype=float) for s in shifts]

    def simulate(self, smiles: list[str]) -> list[np.ndarray | None]:
        if not smiles:
            return []
        out: list[np.ndarray | None] = []
        for start in range(0, len(smiles), self.batch_size):
            batch = smiles[start : start + self.batch_size]
            try:
                out.extend(self._post(batch))
            except Exception as error:  # a dead simulator must not kill the search
                logger.warning(f"Shift simulator failed for a batch of {len(batch)}: {error}")
                out.extend([None] * len(batch))
        return out
