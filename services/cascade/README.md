# CASCADE-2.0 shift prediction service

Wraps [CASCADE-2.0](https://github.com/asbhd/CASCADE-2.0)
(`Predict_SMILES_FF_GPR`) as a 13C shift simulator for `secs.elucidation`.
Its own container because CASCADE pins Python 3.10 + TensorFlow 2.11, which
cannot coexist with SECS (Python 3.11-3.12 + PyTorch).

## Run

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

One entry per molecule, in input order. `null` = could not be predicted,
`[]` = no carbon. `uncertainty` is the 95% half-width in ppm from the GPR
head. Shifts are ordered by RDKit atom index over the carbons.

```python
from secs.elucidation import HttpShiftSimulator, SimulatedShiftVerifier

simulator = HttpShiftSimulator("http://cascade:7997", modality="c_nmr")
verifier = SimulatedShiftVerifier(simulator, observed=peaks, tolerance_ppm=5.0)
```

## Notes

- Geometry-based: each SMILES is embedded with ETKDGv3 and MMFF-optimised.
  That step, not the network, dominates latency -- cache by canonical SMILES
  if you re-score the same molecules.
- Strained bridged systems ([2.2]paracyclophanes, ~2% of chemotion) are
  infeasible for ETKDG's torsion terms. Those fall back to plain distance
  geometry, which embeds them instantly and predicts to ~1.6 ppm rather than
  returning nothing. `CASCADE_FALLBACK_GEOMETRY=0` disables it.
- Predictions are de-standardised with the upstream notebook's constants
  (`x * 50.484337 + 99.798111`), and run on CPU, as upstream does.
- Reported accuracy is ~0.73 ppm against experimental 13C shifts. Measured
  here at 1.71 ppm mean / 0.68 median nearest-peak error on 49 chemotion
  molecules; see `scripts/benchmark_shift_simulators.py`.
