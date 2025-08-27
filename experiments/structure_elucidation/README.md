<div align="center">

# spec2struct

</div>

The results for all individual runs for a specific model and molecules are provided in `luc_subset_parf_pred_42`.
This represents an example based on the PARF model and the molecules from the in-house dataset (34 molecules).
To process the results, one needs to run the following:

```bash
python analyse_pipe.py luc_subset_parf_pred_42 --n_jobs 4
```

This will output a file named `results_luc_subset_parf_pred_42.pkl`.

All figures code are in `aggregate.ipynb`.
