# HOSE shift prediction service

Serves the HOSE lookup table (`secs.elucidation.verifiers.hose`) behind the
same HTTP contract as the cascade and csp5 services, so all three can be
compared or swapped without touching the search code.

## Build and run

The build context is the repository root, so the lookup module can be copied
out of the package instead of duplicated:

```bash
docker build -f services/hose/Dockerfile -t secs/hose .
docker run --rm -p 7995:7995 \
    -v "$PWD/experiments/hose_table.json:/app/hose_table.json:ro" \
    secs/hose
curl -s localhost:7995/health
```

The table is data rather than code and is mounted at run time; build it with
`scripts/build_hose_table.py` from NMRShiftDB.

## API

```bash
curl -s -X POST localhost:7995/ -H 'content-type: application/json' \
     -d '{"smiles": ["CCO", "c1ccccc1"]}'
```

```json
{"shifts":      [[18.1, 57.9], [128.3, ...]],
 "uncertainty": [null, null],
 "n_carbons":   [2, 6],
 "n_predicted": [2, 6]}
```

One entry per input molecule, in input order. `null` means the molecule could
not be parsed, or no carbon matched any environment in the table; `[]` means
it contains no carbon. `uncertainty` is always null.

`n_carbons` and `n_predicted` are specific to this service and exist because
coverage is the thing to watch: a lookup table answers only for environments
it has seen, and a molecule where it answers for 3 of 12 carbons will post a
flattering MAE that means very little. Compare MAEs against the neural
services only alongside these counts.

## Use from SECS

```python
from secs.elucidation import HttpShiftSimulator, SimulatedShiftVerifier

simulator = HttpShiftSimulator("http://hose:7995", modality="c_nmr")
verifier = SimulatedShiftVerifier(simulator, observed=peaks, tolerance_ppm=5.0)
```

In-process use needs no service at all -- `HoseShiftSimulator` wraps the same
table directly, and is the better choice inside a tight search loop.

## Notes

- `HOSE_MIN_COUNT` overrides the table's own `min_count`, raising the number
  of observations an environment needs before it is trusted. Higher values
  trade coverage for accuracy.
- No geometry and no network: prediction is graph traversal plus a hash
  lookup, which is why this is the cheap first stage in front of the
  neural predictors.
