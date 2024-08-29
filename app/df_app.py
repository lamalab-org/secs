# import contextlib
import os

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw


# Function to save RDKit molecule images
def save_molecule_image(smiles, file_path):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, width=600)
    img.save(file_path)
    return file_path


# Streamlit app title
st.title("Molecule Similarity Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Show the dataframe
    st.subheader("DataFrame")
    # nice similarity metric names
    similarity_metric_names = {
        "canonical_smiles": "SMILES",
        "ir_similarity": "IR Similarity",
        "cnmr_similarity": "C-NMR Similarity",
        "hnmr_similarity": "H-NMR Similarity",
        "similarity": "Similarity",
        "sum_of_similarities": "Sum of Similarities",
        "cnmr_ir_similarity": "C-NMR IR Similarity",
        "ir_hnmr_similarity": "H-NMR IR Similarity",
        "cnmr_hnmr_similarity": "C-NMR H-NMR Similarity",
        "tanimoto": "Tanimoto Similarity",
        "unique_hydrogens": "Unique Hydrogens",
        "unique_carbons": "Unique Carbons",
        "sascore": "Synthetic Accessibility Score",
    }
    # rename columns
    data = data.rename(columns=similarity_metric_names)
    data = data.drop(columns=["smiles"])
    st.write(data)
    # User inputs for filtering by unique hydrogens and carbons
    min_hydrogens = st.slider(
        "Minimum Unique Hydrogens:",
        min_value=int(data["Unique Hydrogens"].min()),
        max_value=int(data["Unique Hydrogens"].max()),
        value=int(data["Unique Hydrogens"].min()),
    )
    max_hydrogens = st.slider(
        "Maximum Unique Hydrogens:",
        min_value=int(data["Unique Hydrogens"].min()),
        max_value=int(data["Unique Hydrogens"].max()),
        value=int(data["Unique Hydrogens"].max()),
    )
    min_carbons = st.slider(
        "Minimum Unique Carbons:",
        min_value=int(data["Unique Carbons"].min()),
        max_value=int(data["Unique Carbons"].max()),
        value=int(data["Unique Carbons"].min()),
    )
    max_carbons = st.slider(
        "Maximum Unique Carbons:",
        min_value=int(data["Unique Carbons"].min()),
        max_value=int(data["Unique Carbons"].max()),
        value=int(data["Unique Carbons"].max()),
    )

    # col1, col2, col3 = st.columns([1, 3, 1])
    # with col1:
    #     st.write("Not synthetically accessible")
    # with col2:
    #     synthetic_quantile = st.slider(
    #         "Synthetic Accessibility Quantile:",
    #         min_value=0.0,
    #         max_value=1.0,
    #         value=0.85,
    #     )
    # with col3:
    #     st.write("Synthetically accessible")
    tanimoto_1_df = data[data["Tanimoto Similarity"] == 1]

    # with contextlib.suppress(Exception):
        # Apply filters
    df_filtered = data[
        (data["Unique Hydrogens"] >= min_hydrogens)
        & (data["Unique Hydrogens"] <= max_hydrogens)
        & (data["Unique Carbons"] >= min_carbons)
        & (data["Unique Carbons"] <= max_carbons)
        # & (data["Synthetic Accessibility Score"] > data["Synthetic Accessibility Score"].quantile(1 - synthetic_quantile))
    ]

    # Filter top 5 candidates based on a chosen similarity metric
    similarity_metric = st.selectbox(
        "Select similarity metric to filter by:",
        (
            "IR Similarity",
            "C-NMR Similarity",
            "H-NMR Similarity",
            "Similarity",
            "Sum of Similarities",
            "C-NMR IR Similarity",
            "H-NMR IR Similarity",
            "C-NMR H-NMR Similarity",
            "Tanimoto Similarity",
        ),
    )

    top_n = st.number_input(
        "Select number of top candidates:",
        min_value=1,
        max_value=len(df_filtered),
        value=5,
    )
    # if length of df_filtered is less than top_n, set top_n to length of df_filtered
    top_n = min(top_n, len(df_filtered))
    filtered_df = df_filtered.sort_values(
        by=similarity_metric, ascending=False
    ).head(top_n)

    # Find the molecule with Tanimoto similarity of 1

    # Directory to save images
    image_dir = "molecule_images"
    os.makedirs(image_dir, exist_ok=True)

    # with contextlib.suppress(Exception):
    if not tanimoto_1_df.empty:
        st.subheader("Correct molecule from spectra")
        for index, row in tanimoto_1_df.iterrows():
            image_path = os.path.join(image_dir, f"tanimoto_1_{index}.png")
            save_molecule_image(row["SMILES"], image_path)
            st.image(image_path)
    else:
        st.subheader(
            "No molecule with Tanimoto Similarity of 1 found in the dataset."
        )

    # Display top candidates
    st.subheader(f"Top {top_n} Candidates by {similarity_metric}")

    # display images in columns
    col1, col2, col3 = st.columns(3)
    # assign columns per image
    nr_of_rows = top_n // 3 + 1
    column_list = []
    for _ in range(nr_of_rows):
        column_list.extend([col1, col2, col3])
    for (index, row), col in zip(filtered_df.iterrows(), column_list):
        with col:
            image_path = os.path.join(image_dir, f"mol_{index}.png")
            save_molecule_image(row["SMILES"], image_path)
            st.image(
                image_path,
                caption=f"SMILES:\n{row['SMILES']}\n{similarity_metric}: {row[similarity_metric]:.2f}\nTanimoto Similarity: {row['Tanimoto Similarity']:.2f}",
                use_column_width=True,
            )

else:
    st.write("Please upload a CSV file.")
