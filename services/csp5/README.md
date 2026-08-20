# CSP5 shift prediction service

Wraps [CSP5](https://github.com/Goodman-lab/CSP5) (Goodman lab, trained on
~2.5M experimental NMR spectra) as an HTTP service with the same contract as
the cascade service, so `secs.elucidation` can use either interchangeably.

## Build and run

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

One entry per input molecule, in input order. `null` means the molecule could
not be parsed or predicted; `[]` means it contains no carbon. Shifts are
ordered by RDKit atom index over the carbon atoms. `uncertainty` is always
null (the default CSP5 model has no uncertainty head; the CSP5q quantile
model could supply one later).

## Use from SECS

```python
from secs.elucidation import HttpShiftSimulator, SimulatedShiftVerifier

simulator = HttpShiftSimulator("http://csp5:7996", modality="c_nmr")
verifier = SimulatedShiftVerifier(simulator, observed=peaks, tolerance_ppm=5.0)
```

## Notes

- CSP5 embeds each SMILES itself (ETKDG + force field) before prediction; no
  conformer input is needed and the bundled `CSP5-13C` weights ship inside
  the pip package (~18 MB), so the container needs no model download.
- Paper: [CSP5: Large-scale Neural Chemical Shift Prediction from 2.5 Million
  Experimental NMR Spectra](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15001823/v1);
  reported 13C MAE 0.61 ppm on the assigned Exp22K test set.
