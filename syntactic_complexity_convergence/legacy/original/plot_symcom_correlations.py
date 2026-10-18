from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).parent
RES_DIR = BASE_DIR / "res"
SUMMARY_PATH = RES_DIR / "symcom_correlation_summary.csv"
OUTPUT_DIR = RES_DIR / "symcom_correlation_plots"
METRICS = [
    "syn_value",
    "dependency_length",
    "tree_depth",
    "complex_dependency_ratio",
]
PAIR_TITLES = {
    "syn_value__dependency_length": "Syn ↔ DepLen",
    "syn_value__tree_depth": "Syn ↔ TreeDepth",
    "syn_value__complex_dependency_ratio": "Syn ↔ CDR",
    "dependency_length__tree_depth": "DepLen ↔ TreeDepth",
    "dependency_length__complex_dependency_ratio": "DepLen ↔ CDR",
    "tree_depth__complex_dependency_ratio": "TreeDepth ↔ CDR",
}
HEATMAP_CMAP = sns.color_palette("rocket_r", as_cmap=True)
GRID_COLUMNS = 4
ANNOTATE_CELLS = False
GRID_LABEL_REFERENCE = "A2_follower_symcom"
CUSTOM_LABEL_REPLACEMENTS = {
    "Llm": "LLM",
    "Combined Candor": "Candor",
    "A2": "Mundex",
}


def clean_dataset_label(path_str: str) -> str:
    path = Path(path_str)
    name = path.stem if path.suffix else path.name
    return name


def pretty_label(raw_label: str) -> str:
    label = raw_label.replace("_symcom", "").replace("_", " ")
    label = label.title()
    for old, new in CUSTOM_LABEL_REPLACEMENTS.items():
        label = label.replace(old, new)
    return label


def build_correlation_matrix(row: pd.Series) -> pd.DataFrame:
    matrix = pd.DataFrame(
        np.eye(len(METRICS)), index=METRICS, columns=METRICS, dtype=float
    )
    for col in row.index:
        if "__" not in col:
            continue
        a, b = col.split("__", 1)
        value = row[col]
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value
    return matrix


def configure_matplotlib() -> None:
    sns.set_theme(context="talk", style="white")
    mpl.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
        }
    )


def plot_heatmaps(summary: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in summary.iterrows():
        raw_label = clean_dataset_label(row["file"])
        display_label = pretty_label(raw_label)
        corr_matrix = build_correlation_matrix(row)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            corr_matrix,
            vmin=0,
            vmax=1,
            cmap=HEATMAP_CMAP,
            annot=ANNOTATE_CELLS,
            fmt=".2f",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.7, "pad": 0.02},
            ax=ax,
        )
        ax.set_title(f"{display_label}  $r^2$")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        output_path = OUTPUT_DIR / f"{raw_label}_heatmap.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)


def plot_heatmap_grid(summary: pd.DataFrame) -> None:
    matrices = []
    labels = []
    raw_labels = []
    for _, row in summary.iterrows():
        raw_label = clean_dataset_label(row["file"])
        matrices.append(build_correlation_matrix(row))
        labels.append(pretty_label(raw_label))
        raw_labels.append(raw_label)

    if not matrices:
        return

    columns = GRID_COLUMNS
    rows = math.ceil(len(matrices) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.4, rows * 3.4))
    axes = np.array(axes).reshape(rows, columns)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    for ax in axes.ravel():
        ax.set_visible(False)

    for matrix, label, raw_label, ax in zip(
        matrices, labels, raw_labels, axes.ravel()
    ):
        ax.set_visible(True)
        sns.heatmap(
            matrix,
            vmin=0,
            vmax=1,
            cmap=HEATMAP_CMAP,
            annot=ANNOTATE_CELLS,
            fmt=".2f",
            cbar=False,
            square=True,
            linewidths=0.4,
            annot_kws={"fontsize": 7} if ANNOTATE_CELLS else None,
            ax=ax,
        )
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        if raw_label == GRID_LABEL_REFERENCE:
            ax.set_xlabel("Metric", labelpad=8)
            ax.set_ylabel("Metric", labelpad=8)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
    cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("$r^2$", labelpad=4)
    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.12,
        wspace=0.25,
        hspace=0.35,
    )
    grid_path = OUTPUT_DIR / "symcom_heatmap_grid.png"
    fig.savefig(grid_path, dpi=300)
    plt.close(fig)


def plot_pair_matrix(summary: pd.DataFrame) -> None:
    melted = summary.melt(id_vars="file", var_name="pair", value_name="r_squared")
    melted = melted[melted["pair"].str.contains("__")].copy()
    melted["dataset"] = melted["file"].apply(clean_dataset_label)
    melted["dataset_display"] = melted["dataset"].apply(pretty_label)
    melted["pair_title"] = melted["pair"].map(PAIR_TITLES)

    ordering = (
        melted.groupby("dataset_display")["r_squared"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    pair_order = list(PAIR_TITLES.values())

    matrix = (
        melted.pivot_table(
            index="dataset_display",
            columns="pair_title",
            values="r_squared",
            aggfunc="mean",
        )
        .reindex(index=ordering, columns=pair_order)
    )

    fig, ax = plt.subplots(
        figsize=(max(6, 0.45 * len(pair_order)), 0.35 * len(ordering) + 1)
    )
    sns.heatmap(
        matrix,
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        annot=ANNOTATE_CELLS,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Metric pair")
    ax.set_ylabel("Dataset")
    ax.set_title("Pairwise $r^2$ across datasets")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "symcom_pairwise_heatmap.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH)
    required_cols: Iterable[str] = ["file", *PAIR_TITLES.keys()]
    missing = [col for col in required_cols if col not in summary.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    configure_matplotlib()
    plot_heatmaps(summary)
    plot_heatmap_grid(summary)
    plot_pair_matrix(summary)
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
