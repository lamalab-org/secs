from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def find_original_smiles_from_file_name(file_name):
    return file_name.split("_")[-1]


def determine_if_the_molecule_with_sim_1_in_top_n(sorted_df, n, return_average=False):
    tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
    return average_tanimoto_of_top_n(sorted_df, n) if return_average else int(1 in tanimoto_top_n.to_list())


def average_tanimoto_of_top_n(sorted_df, n):
    tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
    return tanimoto_top_n.max()


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
        # "molbind_similarity": "MolBind Similarity",
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
    # Add more to the UI
    # 1. Read all result files
    result_files = glob("results_img/*.csv")
    result_files = [Path(file) for file in result_files]
    # smiles_from_file = [find_original_smiles_from_file_name(file) for file in result_files]

    to_collect_columns = [
        # "MolBind Similarity",
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

    read_all_dataframe = [pd.read_csv(file) for file in result_files]
    st.write(f"All files: {len(read_all_dataframe)}")
    # filter if Tanimoto 1 not in column
    read_all_dataframe = [df for df in read_all_dataframe if len(df[df["tanimoto"] == 1]) == 1]
    read_all_dataframe = [df for df in read_all_dataframe if len(df) > 10 * top_n]
    # select box for metric to calculate
    average_or_not = st.checkbox(f"Maximum Tanimoto similarity from Top-{top_n}", value=False)
    # select box for adding nmr peaks or not
    add_nmr_peaks = st.checkbox("Add NMR Peaks", value=False)
    # plot molecule size vs performance for each metric
    # plot metrics vs how many molecules are in each csv file
    plot_metrics_vs_number_of_isomers = st.checkbox("Plot Metrics vs Number of Isomers", value=False)
    smiles_list = []

    if add_nmr_peaks:
        # filter all dataframes to have Unique Hydrogens and Unique Carbons the same as the Tanimoto Similarity equal to 1
        for id_, dataframe in enumerate(read_all_dataframe):
            tanimoto_1 = dataframe[dataframe["tanimoto"] == 1]
            smiles_list.append(tanimoto_1["canonical_smiles"].to_list()[0])
            nr_of_unique_hydrogens = tanimoto_1["unique_hydrogens"].to_list()[0]
            nr_of_unique_carbons = tanimoto_1["unique_carbons"].to_list()[0]
            filtered_dataframe = dataframe[
                (dataframe["unique_hydrogens"] == nr_of_unique_hydrogens) & (dataframe["unique_carbons"] == nr_of_unique_carbons)
            ]
            read_all_dataframe[id_] = filtered_dataframe
    else:
        for dataframe in read_all_dataframe:
            tanimoto_1 = dataframe[dataframe["tanimoto"] == 1]
            smiles_list.append(tanimoto_1["canonical_smiles"].to_list()[0])
    st.write(f"Files satisfying the conditions: {len(read_all_dataframe)}")
    # collect metrics for all files
    metrics_list = [collect_statistics_from_files(df, to_collect_columns, top_n, average_or_not) for df in read_all_dataframe]
    # plot len(molecules) vs performance for each metric
    import seaborn as sns

    # white background
    sns.set_style("whitegrid")
    # no grid
    sns.set_style("whitegrid", {"axes.grid": False})

    smiles_list_lengths = [len(smiles) for smiles in smiles_list]

    metrics_df = pd.DataFrame(metrics_list)
    # for column in [*metrics]:
    #     st.write(f"Column: {column}")
    # change index to smiles
    # st.write(metrics_df)
    mean = pd.DataFrame(metrics_df.mean(axis=0))
    mean = mean.rename(columns={0: f"Top {top_n}"})
    # sort by mean
    mean = mean.sort_values(by=f"Top {top_n}", ascending=False)
    # highlight the best method

    if not average_or_not:
        st.subheader(f"Is in Top-{top_n}")
        st.write(mean.T.sort_values(by=f"Top {top_n}", axis=1, ascending=False).style.highlight_max(axis=1))

    if average_or_not:
        st.subheader(f"Average Max Tanimoto Similarity of Top-{top_n}")
        st.write(mean.T.sort_values(by=f"Top {top_n}", axis=1, ascending=False).style.highlight_max(axis=1))

    len_df = [len(df) for df in read_all_dataframe]

    if plot_metrics_vs_number_of_isomers:
        # plot metrics vs how many molecules are in each csv file
        # add comment
        import plotly.express as px

        st.subheader("Plotting metrics vs number of isomers for each file")
        fig = px.scatter(
            metrics_df,
            x=len_df,
            y=to_collect_columns,
            size=smiles_list_lengths,
            # labels={"variable": "Metric", "value": "Max Tanimoto Similarity in Top-5" if average_or_not else f"Is in Top-{top_n}"},
            title="Metrics vs Number of Isomers",
            facet_col="variable",
            facet_col_wrap=2,
            facet_col_spacing=0.05,
            facet_row_spacing=0.05,
            height=1200,
            width=1300,
        )
        # Remove individual x-axis and y-axis titles
        fig.update_xaxes(showticklabels=True, title_text=None)
        fig.update_yaxes(showticklabels=True, title_text=None)
        # one x-axis title for all subplots
        st.plotly_chart(fig)


if __name__ == "__main__":
    app()
