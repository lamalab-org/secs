import streamlit as st
from pathlib import Path
from glob import glob
import pandas as pd


def find_original_smiles_from_file_name(file_name):
    return file_name.split("_")[-1]


def determine_if_the_molecule_with_sim_1_in_top_n(sorted_df, n, return_average=False):
    tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
    return average_tanimoto_of_top_n(sorted_df, n) if return_average else int(1 in tanimoto_top_n.to_list())


def average_tanimoto_of_top_n(sorted_df, n):
    tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
    return tanimoto_top_n.mean()


def collect_statistics_from_files(data, to_collect_columns, n, return_average=False):
    # data = pd.read_csv(file)
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
    # drop duplicates
    data = data.drop_duplicates(subset=["SMILES"])
    metrics = {key: None for key in to_collect_columns}

    for column in to_collect_columns:
        metrics[column] = determine_if_the_molecule_with_sim_1_in_top_n(
            data.sort_values(by=column, ascending=False), n, return_average
        )
    return metrics


# def filter_data


def app():
    st.title("Results Visualization")
    st.write("This is the `Results Visualization`")

    # Add more to the UI
    # 1. Read all result files
    result_files = glob("results/*.csv")
    result_files = [Path(file) for file in result_files]
    # smiles_from_file = [find_original_smiles_from_file_name(file) for file in result_files]

    to_collect_columns = [
        "Sum of Similarities",
        "Similarity",
        "IR Similarity",
        "C-NMR Similarity",
        "H-NMR Similarity",
        "C-NMR IR Similarity",
        "H-NMR IR Similarity",
        "C-NMR H-NMR Similarity",
    ]

    top_n = st.number_input(
        "Select number of top candidates:",
        min_value=1,
        max_value=20,
        value=5,
    )
    # metrics = {key: None for key in to_collect_columns}
    read_all_dataframe = [pd.read_csv(file) for file in result_files]
    # select box for metric to calculate
    average_or_not = st.checkbox("Average of top n", value=False)
    # select box for adding nmr peaks or not
    add_nmr_peaks = st.checkbox("Add NMR Peaks", value=False)
    smiles_list = []
    if add_nmr_peaks:
        # filter all dataframes to have Unique Hydrogens and Unique Carbons the same as the Tanimoto Similarity equal to 1
        for id_, dataframe in enumerate(read_all_dataframe):
            tanimoto_1 = dataframe[dataframe["tanimoto"] == 1]
            smiles_list.append(tanimoto_1["canonical_smiles"].to_numpy()[0])
            nr_of_unique_hydrogens = tanimoto_1["unique_hydrogens"].to_numpy()[0]
            nr_of_unique_carbons = tanimoto_1["unique_carbons"].to_numpy()[0]
            filtered_dataframe = dataframe[
                (dataframe["unique_hydrogens"] == nr_of_unique_hydrogens) & (dataframe["unique_carbons"] == nr_of_unique_carbons)
            ]
            read_all_dataframe[id_] = filtered_dataframe

    # collect metrics for all files
    metrics_list = [collect_statistics_from_files(df, to_collect_columns, top_n, average_or_not) for df in read_all_dataframe]
    # plot len(molecules) vs performance for each metric

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
    # for i, column in enumerate(to_collect_columns):
    #     column_values = [metrics[column] for metrics in metrics_list]
    #     ax[i // 2, i % 2].bar(range(len(column_values)), column_values)
    #     ax[i // 2, i % 2].set_title(column)
    #     ax[i // 2, i % 2].set_xticks(range(len(column_values)))
    # st.pyplot(fig)

    metrics_df = pd.DataFrame(metrics_list)
    # for column in [*metrics]:
    #     st.write(f"Column: {column}")
    # change index to smiles
    st.write(metrics_df)
    mean = pd.DataFrame(metrics_df.mean(axis=0))
    mean = mean.rename(columns={0: f"Top {top_n}"})
    # sort by mean
    mean = mean.sort_values(by=f"Top {top_n}", ascending=False)
    # rewrite column names by top-n
    st.write(mean)


if __name__ == "__main__":
    app()
