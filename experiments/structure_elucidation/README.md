<div align="center">

# Spec2Struct

</div>

All the files necessary for this Notebook to run can be found on [Zenodo](https://zenodo.org/records/14177705).

## ⚙️System Requirements

To run the notebook you need a CUDA compatible device.

## 📓 Demo

The `demo.ipynb` is a Jupyter Notebook showcasing the pipeline that consists of 4 steps. It is assumed that we know the molecular formula of the compound. All the files for this demo can be found on the

  1. Preprocess and embed the spectra you use to idenfify the correct molecule:
     - 13C NMR is a binary vector of length 512 covering the range of chemical shifts: 0 to 300 ppm. The indices represent the chemical shifts and the values are binary values 0 and 1 denoting the presence / absence of the shift.
     - 1H NMR is a vector of length 10,000 covering the range of chemical shifts: -2 to 10 ppm. The indices represent the chemical shifts, while the values present the intensity of the peaks normalized between 0 and 1.
     - IR is a vector of length 1600 covering the range of wavelengths 600 - 3800 $cm^{-1}$. The values in the vector match the transmittance that can range between 0 and 1.
  2. Based on a list of generated molecular formulas (based on the correct molecular formula) we can extract a list of SMILES from PubChem.
     - We embed all the SMILES using three different contrastively trained models
        - IR-SMILES model (checkpoint at experiments/checkpoints/ir_simulated_large_dataset_20241015_2122/best_model.ckpt)
        - 13C NMR - SMILES (checkpoint at experiments/checkpoints/cnmr_simulated_large_dataset_20241016_1008/best_model.ckpt)
        - 1H NMR - SMILES (checkpoint at experiments/checkpoints/hnmr_simulated_detailed_cnn_architecture_large_dataset_20241015_2122/best_model.ckpt)
  3.  Sort SMILES based on the sum of similarities between the SMILES embedding and the respective spectra embeddings: $\frac{1}{3} \sum_{x \in \{\text{IR}, ^{1}H, ^{13}C\}}d_{\text{cos}}\left( \varepsilon_{x}, \varepsilon_{\text{SMILES-}x} \right)$
  4. Use the best `N` SMILES from step (3.) to start a genetic algorithm run.
