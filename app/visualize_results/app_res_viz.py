from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from loguru import logger


def find_original_smiles_from_file_name(file_name):
    return file_name.split("_")[-1]


def determine_if_the_molecule_with_sim_1_in_top_n(
    sorted_df: pd.DataFrame, n: int, return_average: bool = False, exclude_original_molecule: bool = True
):
    tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
    return (
        average_tanimoto_of_top_n(sorted_df, n, exclude_original_molecule=exclude_original_molecule)
        if return_average
        else int(1 in tanimoto_top_n.to_list())
    )


def average_tanimoto_of_top_n(sorted_df, n, exclude_original_molecule: bool = True):
    # return tanimoto_top_n.mean()
    # remove the 1 tanimoto similarity
    if exclude_original_molecule:
        sorted_df = sorted_df[sorted_df["Tanimoto Similarity"] != 1]
        tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
        # max_possible = sorted_df[sorted_df["Tanimoto Similarity"] != 1]["Tanimoto Similarity"].max()
        return tanimoto_top_n.max()
    tanimoto_top_n = sorted_df.head(n)["Tanimoto Similarity"]
    return tanimoto_top_n.max()


def collect_statistics_from_files(data, to_collect_columns, n, return_average=False, exclude_original_molecule=True):
    # data = pd.read_csv(file)
    # nice similarity metric names
    similarity_metric_names = {
        "canonical_smiles": "SMILES",
        "ir_similarity": "IR Similarity",
        "cnmr_similarity": "C-NMR Similarity",
        "hnmr_similarity": "H-NMR Similarity",
        "similarity": "Similarity of sums",
        "sum_of_similarities": "Sum of Similarities",
        "cnmr_ir_similarity": "C-NMR IR Similarity",
        "ir_hnmr_similarity": "H-NMR IR Similarity",
        "cnmr_hnmr_similarity": "C-NMR H-NMR Similarity",
        "tanimoto": "Tanimoto Similarity",
        "unique_hydrogens": "Unique Hydrogens",
        "unique_carbons": "Unique Carbons",
        "sascore": "Synthetic Accessibility Score",
        "multi_spec_similarity": "MultiSpec Similarity",
        # "molbind_similarity": "MolBind Similarity",
    }

    # rename columns
    data = data.rename(columns=similarity_metric_names)
    # drop duplicates
    data = data.drop_duplicates(subset=["SMILES"])
    metrics = {key: None for key in to_collect_columns}

    for column in to_collect_columns:
        metrics[column] = determine_if_the_molecule_with_sim_1_in_top_n(
            data.sort_values(by=column, ascending=False), n, return_average, exclude_original_molecule
        )
    return metrics


def app():
    st.title("Results Visualization")
    # Add more to the UI
    # 1. Read all result files
    result_files = Path("../../experiments/structure_elucidation/results").rglob("*.csv")
    # smiles_from_file = [find_original_smiles_from_file_name(file) for file in result_files]

    to_collect_columns = [
        # "MolBind Similarity",
        "Sum of Similarities",
        "Similarity of sums",
        "MultiSpec Similarity",
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
        value=1,
    )

    read_all_dataframe = [pd.read_csv(file) for file in result_files]
    st.write(f"All files: {len(read_all_dataframe)}")
    # filter if Tanimoto 1 not in column
    read_all_dataframe = [df for df in read_all_dataframe if len(df[df["tanimoto"] == 1]) == 1]
    read_all_dataframe = [df for df in read_all_dataframe if len(df) > 10 * top_n]
    # select box for metric to calculate
    average_or_not = st.checkbox(f"Maximum Tanimoto similarity from Top-{top_n}", value=False)
    exclude_button = st.checkbox("Exclude original molecule from the top candidates", value=False)

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
    metrics_list = [
        collect_statistics_from_files(df, to_collect_columns, top_n, average_or_not, exclude_button) for df in read_all_dataframe
    ]
    # print maximum possible score
    # second best score in all dataframes
    # plot len(molecules) vs performance for each metric
    smiles_list_lengths = [len(smiles) for smiles in smiles_list]

    metrics_df = pd.DataFrame(metrics_list)
    # for column in [*metrics]:
    #     st.write(f"Column: {column}")
    # change index to smiles
    # st.write(metrics_df)
    mean = pd.DataFrame(metrics_df.mean(axis=0))
    mean = mean.rename(columns={0: f"Top {top_n}"})
    # sort by mean
    # mean = mean.sort_values(by=f"Top {top_n}", ascending=False)
    # st.write(f"Max possible score: {max_possible}")
    col1, col2 = st.columns(2)
    if not average_or_not:
        fig = px.bar(
            mean,
            x=mean.index,
            y=f"Top {top_n}",
            color=mean.index,
            labels={"index": "Metric", f"Top {top_n}": f"Is in Top-{top_n}"},
            height=600,
            width=800,
        )
        st.plotly_chart(fig)
    if average_or_not:
        # use plotly to plot the bar chart
        fig = px.bar(
            mean,
            x=mean.index,
            y=f"Top {top_n}",
            color=mean.index,
            labels={"index": "Metric", f"Top {top_n}": f"Average Max Tanimoto Similarity of Top-{top_n}"},
            height=600,
            width=800,
        )
        st.plotly_chart(fig)

    len_df = [len(df) for df in read_all_dataframe]

    if plot_metrics_vs_number_of_isomers:
        # plot metrics vs how many molecules are in each csv file
        # add comment

        st.subheader("Plotting metrics vs number of isomers for each file")
        # set a plotly theme
        px.defaults.template = "plotly_white"
        # add boxplot on the side
        fig = px.scatter(
            metrics_df,
            x=len_df,
            y=to_collect_columns,
            size=smiles_list_lengths,
            # labels={"variable": "Metric", "value": "Max Tanimoto Similarity in Top-5" if average_or_not else f"Is in Top-{top_n}"},
            facet_col="variable",
            facet_col_wrap=3,
            facet_col_spacing=0.05,
            facet_row_spacing=0.05,
            hover_name=smiles_list,
            height=600,
            width=900,
        )

        # add a line with average performance for each metric
        # add average performance to each subplot from the mean dataframe
        # scale font size
        fig.update_layout(font={"size": 10})
        # add one x-axis title for all subplots
        fig.update_xaxes(title_text="Number of isomers")
        # add performance on the plot
        # add annotation with average performance as interrupted line
        # tight layout
        fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
        # align legend properly
        # all plots same axis limits
        fig.update_yaxes(range=[0, 1])
        # add legend with size meaning
        fig.update_layout(legend_title_text="Similarity metric")
        # remove subplots titles
        fig.for_each_annotation(lambda a: a.update(text=""))
        # align the legend entries in 3 columns
        fig.update_layout(legend={"itemsizing": "constant"})
        # Remove individual x-axis and y-axis titles
        fig.update_xaxes(showticklabels=True, title_text=None)
        fig.update_yaxes(showticklabels=True, title_text=None)
        # save the plot to pdf
        fig.write_image("metrics_vs_isomers.pdf", scale=3)
        # one x-axis title for all subplots
        st.plotly_chart(fig)

    plot_histograms = st.checkbox("Plot histograms of metrics", value=False)
    if plot_histograms:
        # historgrams in subplots with kde
        st.subheader("Histograms of metrics")
        fig = px.histogram(
            metrics_df.melt(),
            x="value",
            facet_col="variable",
            facet_col_wrap=3,
            facet_col_spacing=0.05,
            facet_row_spacing=0.1,
            height=600,
            width=800,
            nbins=20,
            # same colors as in scatter plot
            color_discrete_sequence=px.colors.qualitative.Plotly,
            histnorm="percent",
            barmode="overlay",
            # kde=True,
        )
        # horizontal histograms
        st.plotly_chart(fig)

    # allow specific smiles to be selected

    smiles_dataframe_dict = dict(zip(smiles_list, read_all_dataframe))

    selected_smiles = st.selectbox("Select SMILES", smiles_list)
    selected_df = smiles_dataframe_dict[selected_smiles]
    # collect metrics for the selected smiles
    # plot tanimoto similarity vs other metrics for the selected smiles
    if selected_df is not None:
        st.write(selected_df)
        # divide sum of similarities by the maximum possible
        selected_df["sum_of_similarities"] = selected_df["sum_of_similarities"] / 3
        # plot tanimoto similarity vs other metrics
        fig = px.scatter(
            selected_df,
            y="tanimoto",
            x=[
                "ir_similarity",
                "cnmr_similarity",
                "hnmr_similarity",
                "multi_spec_similarity",
                "similarity",
                "sum_of_similarities",
            ],
            hover_name="canonical_smiles",
            height=600,
            width=800,
            labels={"variable": "Metric", "value": "Similarity"},
        )
        # R2 score for each metric
        # print R2 score for each metric
        logger.debug(f"R2 score for each metric: {selected_df[[
                "ir_similarity",
                "cnmr_similarity",
                "hnmr_similarity",
                "multi_spec_similarity",
                "similarity",
                "sum_of_similarities",
                "tanimoto",
            ]].corr()['tanimoto']**2}")
        st.plotly_chart(fig)

    if st.button("Download metrics"):
        metrics_df.to_csv("metrics.csv")
        st.write("Metrics downloaded successfully")


if __name__ == "__main__":
    app()
