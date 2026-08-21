# HOSE shift prediction service

Serves the HOSE lookup table (`secs.elucidation.verifiers.hose`).

## Run

The build context is the repository root, so the lookup module is copied out
of the package rather than duplicated. The table is data, not code, and is
mounted at run time; build it with `scripts/build_hose_table.py`.

```bash
docker build -f services/hose/Dockerfile -t secs/hose .
docker run --rm -p 7995:7995 \
    -v "$PWD/experiments/hose_table.json:/app/hose_table.json:ro" \
    secs/hose
curl -s localhost:7995/health
```

## API

```bash
curl -s -X POST localhost:7995/ -H 'content-type: application/json' \
     -d '{"smiles": ["CCO", "c1ccccc1"]}'
```

```json
{"shifts":      [[14.3, 64.9], [128.59, ...]],
 "uncertainty": [null, null],
 "n_carbons":   [2, 6],
 "n_predicted": [2, 6]}
```

One entry per molecule, in input order. `null` = unparseable, or no carbon
matched any environment in the table; `[]` = no carbon. `uncertainty` is
always null.

`n_carbons` and `n_predicted` are specific to this service, because coverage
is the thing to watch: a table answers only for environments it has seen,
and a molecule where it answers for 3 of 12 carbons posts a flattering MAE
that means little. Read its error alongside these counts.

```python
from secs.elucidation import HttpShiftSimulator, SimulatedShiftVerifier

simulator = HttpShiftSimulator("http://hose:7995", modality="c_nmr")
verifier = SimulatedShiftVerifier(simulator, observed=peaks, tolerance_ppm=5.0)
```

In-process use needs no service -- `HoseShiftSimulator` wraps the same table
directly, and is the better choice inside a tight search loop.

## Notes

- No geometry and no network: prediction is graph traversal plus a hash
  lookup. That makes it ~20x faster than the neural services and the natural
  cheap first stage in front of them.
- `HOSE_MIN_COUNT` overrides the table's `min_count`, raising how many
  observations an environment needs before it is trusted. Higher values
  trade coverage for accuracy.
- Measured at 2.48 ppm mean / 1.44 median nearest-peak error on 49 chemotion
  molecules, with full carbon coverage and no failures -- roughly 4x the
  error of csp5; see `scripts/benchmark_shift_simulators.py`.
