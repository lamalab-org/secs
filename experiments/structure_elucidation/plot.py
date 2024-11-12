from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None


def find_original_smiles_from_file_name(file_name):
    return file_name.split("_")[-1]


def determine_if_the_molecule_with_sim_1_in_top_n(
    sorted_df: pd.DataFrame, n: int, return_average: bool = False, exclude_original_molecule: bool = True
):
    tanimoto_top_n = sorted_df.head(n)["Tanimoto similarity"]
    return (
        average_tanimoto_of_top_n(sorted_df, n, exclude_original_molecule=exclude_original_molecule)
        if return_average
        else int(1 in tanimoto_top_n.to_list())
    )


def average_tanimoto_of_top_n(sorted_df, n, exclude_original_molecule: bool = True):
    if exclude_original_molecule:
        sorted_df = sorted_df[sorted_df["Tanimoto similarity"] != 1]
        tanimoto_top_n = sorted_df.head(n)["Tanimoto similarity"]
        max_possible = sorted_df[sorted_df["Tanimoto similarity"] != 1]["Tanimoto similarity"].max()
        return (max_possible - tanimoto_top_n.max()) / max_possible * 100
    tanimoto_top_n = sorted_df.head(n)["Tanimoto similarity"]
    return tanimoto_top_n.max()


def collect_statistics_from_files(data, to_collect_columns, n, return_average=False, exclude_original_molecule=True):
    similarity_metric_names = {
        "canonical_smiles": "SMILES",
        "ir_similarity": "IR",
        "cnmr_similarity": "¹³C-NMR",
        "hnmr_similarity": "¹H-NMR",
        "cnmr_ir_similarity": "¹³C-NMR + IR",
        "ir_hnmr_similarity": "IR + ¹H-NMR",
        "cnmr_hnmr_similarity": "¹³C-NMR + ¹H-NMR",
        "tanimoto": "Tanimoto similarity",
        "unique_hydrogens": "Unique Hydrogens",
        "unique_carbons": "Unique Carbons",
        "sascore": "Synthetic Accessibility Score",
        "sum_of_similarities": "¹³C-NMR + IR + ¹H-NMR",
    }

    data = data.rename(columns=similarity_metric_names)
    data = data.drop_duplicates(subset=["SMILES"])
    metrics = {key: None for key in to_collect_columns}

    for column in to_collect_columns:
        metrics[column] = determine_if_the_molecule_with_sim_1_in_top_n(
            data.sort_values(by=column, ascending=False), n, return_average, exclude_original_molecule
        )
    return metrics


# Main code without streamlit
result_files = list(
    Path("../../experiments/structure_elucidation/results_large_dataset_multiple_molecular_formulas_per_file").rglob("*.csv")
)
to_collect_columns = [
    "¹³C-NMR + IR + ¹H-NMR",
    "¹³C-NMR + ¹H-NMR",
    "¹³C-NMR + IR",
    "IR + ¹H-NMR",
    "¹³C-NMR",
    "¹H-NMR",
    "IR",
]

top_n = 1  # For example, set top_n to 1
read_all_dataframe = [pd.read_csv(file) for file in result_files]

read_all_dataframe = [df for df in read_all_dataframe if len(df[df["tanimoto"] == 1]) == 1]
read_all_dataframe = [df for df in read_all_dataframe if len(df) > 10 * top_n]

average_or_not = False
exclude_button = False
add_nmr_peaks = False
plot_metrics_vs_number_of_isomers = False
smiles_list = []

for dataframe in read_all_dataframe:
    tanimoto_1 = dataframe[dataframe["tanimoto"] == 1]
    smiles_list.append(tanimoto_1["canonical_smiles"].to_list()[0])

metrics_list = [
    collect_statistics_from_files(df, to_collect_columns, top_n, average_or_not, exclude_button) for df in read_all_dataframe
]

metrics_df = pd.DataFrame(metrics_list)
mean = pd.DataFrame(metrics_df.mean(axis=0))
mean = mean.rename(columns={0: f"Top {top_n}"})

# Plotting the results
if not average_or_not:
    # custom_colors = {
    #     "ref. top 1": "rgba(152, 86, 86, 0.3)",
    #     "top 1": "rgba(152, 86, 86, 0.7)",
    #     "ref. top 5": "rgba(86, 86, 152, 0.3)",
    #     "top 5": "rgba(86, 86, 152, 0.7)",
    #     "in the population": "rgba(86, 86, 86, 0.7)",
    # }
    fig = px.bar(
        mean,
        y=mean.index,
        x=f"Top {top_n}",
        color=mean.index,
        color_discrete_map={
            "¹³C-NMR + IR + ¹H-NMR": "rgba(86, 86, 86, 0.7)",
            "¹³C-NMR + ¹H-NMR": "rgba(86, 86, 152, 0.7)",
            "IR + ¹H-NMR": "rgba(86, 86, 152, 0.7)",
            "¹³C-NMR + IR": "rgba(86, 86, 152, 0.7)",
            "¹H-NMR": "rgba(152, 86, 86, 0.7)",
            "¹³C-NMR": "rgba(152, 86, 86, 0.7)",
            "IR": "rgba(152, 86, 86, 0.7)",
        },
        orientation="h",
    )
    # change margins to 0
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    sizing = 305
    fig.update_layout(
        template="plotly_white",
        width=sizing * 1.618,
        height=sizing,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        plot_bgcolor="white",
        font={"color": "rgb(120, 120, 120)"},
        bargap=0.15,
        bargroupgap=0.03,
        shapes=[
            # Add x-axis line
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": -0.5,
                "y1": -0.5,  # Position below the bottom bar
                "line": {"color": "rgb(120, 120, 120)", "width": 1},
            },
            # Add y-axis line
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": 0,
                "y0": 0,
                "x1": 0,
                "y1": 1,
                "line": {"color": "rgb(120, 120, 120)", "width": 1},
            },
        ],
    )

    # Update x-axis
    fig.update_xaxes(
        title_text="fraction of correct predictions",
        range=[-0.0, 1.02],  # Add padding
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="outside",
        tickwidth=1,
        tickcolor="rgb(120, 120, 120)",
        ticklen=5,
        tickvals=np.arange(0, 1.2, 0.2),
        tickformat=".1f",
    )

    # Update y-axis
    fig.update_yaxes(
        title_text="metric",
        showgrid=False,
        gridwidth=1,
        gridcolor="rgba(173, 216, 230, 0.3)",
        zeroline=False,
        showline=False,
        ticks="outside",
        tickwidth=1,
        tickcolor="rgb(120, 120, 120)",
        ticklen=5,
    )

    # Add reference line for perfect score
    fig.add_vline(
        x=1,
        line_dash="dot",
        line_color="rgb(120, 120, 120)",
        annotation={
            "text": "perfect score",
            "textangle": 90,
            "x": 1.01,
            "yref": "paper",
            "y": 0.7,
            "showarrow": False,
            "font": {"color": "rgb(120, 120, 120)"},
        },
    )
    fig.update_layout(showlegend=False, width=sizing * 1.618, height=sizing, bargap=0.15, bargroupgap=0.03, plot_bgcolor="white")
    fig.update_layout(plot_bgcolor="white")
    fig.update_yaxes(title_text="")
    fig.update_xaxes(range=[0, 1], tickvals=np.arange(0, 1.2, 0.1), title_text="fraction of correctly retrieved @1", tickangle=0)
    for trace in fig.data:
        trace.width = 0.6
    fig.update_layout(font_family="CMU Sans Serif")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # Saving the figure
    fig.write_image(f"is_in_{top_n}.pdf", scale=3)

else:
    fig = px.bar(
        mean,
        x=mean.index,
        y=f"Top {top_n}",
        color=mean.index,
        labels={"index": "Metric", f"Top {top_n}": f"Error of Max Tanimoto Top-{top_n}"},
        height=600,
        width=800,
    )
    # change margins to 0
    fig.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0})
    sizing = 305
    fig.update_layout(showlegend=False, width=sizing * 1.618, height=sizing, bargap=0.15, bargroupgap=0.03, plot_bgcolor="white")
    fig.update_layout(plot_bgcolor="white")
    fig.update_yaxes(title_text="")
    fig.update_xaxes(range=[0, 1], tickvals=np.arange(0, 1.2, 0.1), title_text="fraction of correctly retrieved @1", tickangle=0)
    for trace in fig.data:
        trace.width = 0.4
    fig.update_layout(font_family="CMU Sans Serif")
    fig.update_xaxes(showgrid=False)
    # Saving the figure
    fig.write_image(f"best_{top_n}.pdf", scale=3)

# Additional plotting metrics vs number of isomers
len_df = [len(df) for df in read_all_dataframe]

if plot_metrics_vs_number_of_isomers:
    fig = px.scatter(
        metrics_df,
        x=len_df,
        y=to_collect_columns,
        size=[len(smiles) for smiles in smiles_list],
        facet_col="variable",
        facet_col_wrap=3,
        facet_col_spacing=0.05,
        facet_row_spacing=0.05,
        hover_name=smiles_list,
        height=600,
        width=900,
    )
    fig.update_layout(font={"size": 10})
    fig.update_xaxes(title_text="Number of isomers")
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(legend_title_text="Similarity metric")
    fig.update_layout(legend={"itemsizing": "constant"})
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(showticklabels=True)
