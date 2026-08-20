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

- CSP5 embeds each SMILES itself; the weights ship inside the pip package
  (~18 MB), so the container downloads no model at build time.
- `CSP5_MAX_EMBED_TRIES` defaults to 1, not the library's 20. Molecules
  ETKDG cannot embed fail all 20 attempts identically -- 4s at one try, 69s
  at twenty, `null` either way -- and retries were measured to rescue
  nothing (149/150 predicted at 1, 2 and 20 tries).
- Those molecules are then rescued rather than dropped. ETKDG rejects
  strained bridged systems ([2.2]paracyclophanes, ~2% of chemotion) because
  its torsion terms cannot be satisfied, not because the structure is hard
  in 3D: plain distance geometry embeds them in milliseconds, and after MMFF
  the aromatic decks show the ~0.08 A bend these molecules are known for.
  Prediction then runs from that geometry via `predict_structures`, giving
  ~1.1 ppm where the service previously returned nothing.
  `CSP5_FALLBACK_GEOMETRY=0` disables it.
- Paper: [CSP5](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001823/v1),
  reporting 13C MAE 0.61 ppm on the assigned Exp22K test set. Measured here
  at 1.48 ppm mean / 0.39 median nearest-peak error on 49 chemotion
  molecules, ahead of cascade on 42 of them; see
  `scripts/benchmark_shift_simulators.py`.
