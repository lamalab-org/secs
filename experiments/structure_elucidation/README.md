<div align="center">

## How to run MAYGEN pipeline?

</div>

### :full_moon: Step 1

```sh
java -jar MAYGEN-1.8.jar -f  C4H5ClN4  -v -t -o -smi
```

### :last_quarter_moon: Step 2

```sh
python maygen_out_to_canonical.py C4H5ClN4.smi C4H5ClN4.csv
```

### :new_moon: Step 3

The default value for the contributions of different spectra embeddings is 1.

```sh
python prune.py C4H5ClN4.csv pruned_embeddings.csv 51 --ir-ratio=1 --cnmr_ratio=1 --hnmr_ratio=1 --synthetic_access_quantile=0.01
```
