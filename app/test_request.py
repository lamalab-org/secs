import numpy as np
import pandas as pd
import requests

from molbind.utils.spec2struct import smiles_to_molecular_formula

# Replace with your actual URL
data = pd.read_parquet("../data/nmrshiftdb/hnmr_cnmr_validation_set.parquet")
idx = 103
spectrum = {"y": data.h_nmr.iloc[idx].tolist(), "x": np.linspace(10, -2, len(data.h_nmr.iloc[idx])).tolist()}

mf = smiles_to_molecular_formula(data.smiles.iloc[idx])

print(mf)
# url = "https://lamalab-org--spec2struct-elucidate-spectrum.modal.run"
# response = requests.post(url, json={"mf": mf, "spectrum": spectrum})
