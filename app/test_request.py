import json

import numpy as np
import pandas as pd
import requests

from molbind.utils.spec2struct import smiles_to_molecular_formula

# Replace with your actual URL
# data = pd.read_parquet("/home/mirzaa/projects/MoleculeBind/data/data_chemotion_final_clean_luc/chemotion1500.parquet")
data = pd.read_parquet("angewandte_molecules.parquet")
# idx = 103

for idx in range(0, 10):
    spectrum = {"y": data.h_nmr.iloc[idx].tolist(), "x": np.linspace(10, -2, len(data.h_nmr.iloc[idx])).tolist()}

    mf = smiles_to_molecular_formula(data.smiles.iloc[idx])

    print(idx, mf, data.smiles.iloc[idx])  # noqa: T201
    url = "https://lamalab-org--spec2struct-elucidate-spectrum.modal.run"
    response = requests.post(
        url, json={"mf": mf, "spectrum": spectrum, "pop_ga": 10000, "offspring_ga": 1024, "gens_ga": 10, "model": "regular"}
    )

    with open(f"angewandte/{idx}.json", "w") as f:  # noqa: PTH123
        json.dump(response.json(), f, indent=4)
