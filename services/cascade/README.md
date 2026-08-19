# CASCADE-2.0 shift prediction service

Wraps [CASCADE-2.0](https://github.com/asbhd/CASCADE-2.0) (`Predict_SMILES_FF_GPR`)
as an HTTP service so `secs.elucidation` can use it as a spectrum simulator.

## Why a separate container

CASCADE pins Python 3.10 + TensorFlow 2.11 + KGCNN 2.2.1. SECS runs Python
3.11-3.12 + PyTorch 2.10, and TF 2.11 has no Python 3.12 wheels, so the two
cannot share an environment. This follows the same pattern as the existing
`vectordb` and `forward_synthesis` services.

## Build and run

```bash
docker build -t secs/cascade services/cascade
docker run --rm -p 7997:7997 secs/cascade
curl -s localhost:7997/health
```

## API

```bash
curl -s -X POST localhost:7997/ -H 'content-type: application/json' \
     -d '{"smiles": ["CCO", "c1ccccc1"]}'
```

```json
{"shifts":      [[58.1, 18.4], [128.4]],
 "uncertainty": [[1.2, 1.1],   [0.9]]}
```

One entry per input molecule, in input order. `null` means the molecule could
not be embedded or predicted; `[]` means it contains no carbon. `uncertainty`
is the 95% half-width in ppm, from the model's GPR head.

## Use from SECS

```python
from secs.elucidation import HttpShiftSimulator, SimulatedShiftVerifier

simulator = HttpShiftSimulator("http://cascade:7997", modality="c_nmr")
verifier = SimulatedShiftVerifier(simulator, observed=peaks, tolerance_ppm=5.0)
```

## Notes

- The model is geometry-based: each SMILES is embedded with ETKDGv3 and
  MMFF-optimised before prediction. That conformer step, not the network,
  dominates latency. A GA generation of ~2000 candidates is a real cost, so
  cache by canonical SMILES if you run repeatedly.
- Predictions are de-standardised with the constants from the upstream
  notebook (`x * 50.484337 + 99.798111`).
- Runs on CPU (`CUDA_VISIBLE_DEVICES=-1`), matching the upstream notebook.
- Reported accuracy is ~0.73 ppm against experimental 13C shifts.
