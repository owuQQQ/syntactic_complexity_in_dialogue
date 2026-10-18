from __future__ import annotations

import itertools
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent
RES_DIR = BASE_DIR / "res"
METRICS = [
    "syn_value",
    "dependency_length",
    "tree_depth",
    "complex_dependency_ratio",
]
PAIR_NAMES = [f"{a}__{b}" for a, b in itertools.combinations(METRICS, 2)]
SKIP_KEYWORDS = {"vm2", "texprax"}


def load_metric_frame(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t")
    missing = [col for col in METRICS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {file_path}")
    return df[METRICS].dropna()


def summarize_correlations() -> pd.DataFrame:
    summary_rows = []
    correlation_tables: dict[str, pd.DataFrame] = {}

    for file_path in sorted(RES_DIR.glob("*symcom.txt")):
        lowered = file_path.name.lower()
        if any(keyword in lowered for keyword in SKIP_KEYWORDS):
            continue
        metric_frame = load_metric_frame(file_path)
        corr = metric_frame.corr()
        relative_name = str(file_path.relative_to(BASE_DIR))
        correlation_tables[relative_name] = corr

        row = {"file": relative_name}
        for a, b in itertools.combinations(METRICS, 2):
            row[f"{a}__{b}"] = corr.at[a, b] ** 2
        summary_rows.append(row)

    if not summary_rows:
        raise RuntimeError("No matching symcom files were processed.")

    summary_df = pd.DataFrame(summary_rows)
    output_path = RES_DIR / "symcom_correlation_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"Correlation summary written to {output_path}")
    for name, corr in correlation_tables.items():
        print(f"\n{name}\n{'-' * len(name)}")
        print(corr.round(4))

    return summary_df

if __name__ == "__main__":
    summarize_correlations()
