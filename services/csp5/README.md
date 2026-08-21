# CSP5 shift prediction service

Wraps [CSP5](https://github.com/Goodman-lab/CSP5) (Goodman lab, trained on
~2.5M experimental NMR spectra) with the same contract as the cascade and
hose services, so the three are interchangeable.

## Run

```bash
docker build -t secs/csp5 services/csp5
docker run --rm -p 7996:7996 secs/csp5
curl -s localhost:7996/health
```

## API

```bash
curl -s -X POST localhost:7996/ -H 'content-type: application/json' \
     -d '{"smiles": ["CCO", "c1ccccc1"]}'
```

```json
{"shifts":      [[18.19, 58.14], [128.04, ...]],
 "uncertainty": [null, null]}
```

One entry per molecule, in input order. `null` = could not be predicted,
`[]` = no carbon. Shifts are ordered by RDKit atom index over the carbons.
`uncertainty` is always null -- the default model has no uncertainty head;
the bundled `CSP5q-13C` quantile model could supply one later.

```python
from secs.elucidation import HttpShiftSimulator, SimulatedShiftVerifier

simulator = HttpShiftSimulator("http://csp5:7996", modality="c_nmr")
verifier = SimulatedShiftVerifier(simulator, observed=peaks, tolerance_ppm=5.0)
```

## Notes

- Parallelism & retries: Weights ship in the pip package (~18 MB, no build-time download); requests are sharded over a process pool (CSP5_WORKERS, default 8) because CSP5's own num_workers is broken in 0.2.18 — 106.6s → 14.4s on 800 GA candidates, identical predictions. CSP5_MAX_EMBED_TRIES defaults to 1 since retries rescue nothing (149/150 at 1, 2 and 20 tries) and cost 4s vs 69s.
- Fallback & accuracy: Molecules ETKDG can't embed (strained bridged systems, ~2% of chemotion) fall back to plain distance geometry + MMFF and predict_structures, giving ~1.1 ppm where the service previously returned null (CSP5_FALLBACK_GEOMETRY=0 disables). Measured 1.48 ppm mean / 0.39 median nearest-peak error on 49 chemotion molecules — ahead of cascade on 42 — vs the paper's 0.61 ppm 13C MAE on Exp22K.