from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def calculate_lengths(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Aggregate per-turn token counts and turn counts for each dialogue."""
    frame = pd.read_csv(input_path)
    required = {"dialogue_id", "word_count"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Metrics input is missing {sorted(missing)}. Run calculate_metrics first."
        )
    frame["word_count"] = pd.to_numeric(frame["word_count"], errors="coerce")
    if frame["word_count"].isna().any() or (frame["word_count"] < 0).any():
        raise ValueError("word_count must contain finite nonnegative values.")
    lengths = (
        frame.groupby("dialogue_id", sort=False)
        .agg(word_count=("word_count", "sum"), turn_count=("word_count", "size"))
        .reset_index()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lengths.to_csv(output_path, index=False)
    print(f"Wrote word and turn counts for {len(lengths)} dialogues to {output_path}.")
    return lengths


def main() -> None:
    """Parse CLI arguments and create a dialogue-level length sidecar."""
    parser = argparse.ArgumentParser(
        description="Aggregate Stanza token and turn counts into detection length sidecars."
    )
    parser.add_argument("--input", required=True, type=Path, help="Metrics CSV.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    calculate_lengths(args.input, args.output)


if __name__ == "__main__":
    main()
