import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def add_phase_labels(frame: pd.DataFrame, phase_counts: tuple[int, ...] = (8, 16)) -> pd.DataFrame:
    """Add contiguous, one-based phase labels per dialogue and speaker."""
    required = {"dialogue_id", "speaker", "turn_index"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    if not phase_counts or any(count < 2 for count in phase_counts):
        raise ValueError("Each requested phase count must be at least 2.")

    result = frame.reset_index(drop=True).copy()
    result["turn_index"] = pd.to_numeric(result["turn_index"], errors="raise")
    for count in phase_counts:
        labels = np.zeros(len(result), dtype=int)
        column = f"phase_{count}"
        for _, group in result.groupby(["dialogue_id", "speaker"], sort=False):
            ordered_indices = group.sort_values("turn_index", kind="stable").index.to_numpy()
            for phase_number, positions in enumerate(np.array_split(ordered_indices, count), start=1):
                labels[positions] = phase_number
        if (labels == 0).any():
            raise ValueError(f"Failed to assign all rows to {column}.")
        result[column] = labels
    return result


def main() -> None:
    """Parse CLI arguments and save requested dialogue phase labels."""
    parser = argparse.ArgumentParser(
        description="Add reusable contiguous phase labels per dialogue and speaker."
    )
    parser.add_argument("--input", required=True, type=Path, help="Metrics CSV.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--phases",
        default="8,16",
        help="Comma-separated phase counts (default: 8,16).",
    )
    args = parser.parse_args()
    counts = tuple(int(value.strip()) for value in args.phases.split(",") if value.strip())
    labeled = add_phase_labels(pd.read_csv(args.input), counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(args.output, index=False)
    print(f"Wrote {len(labeled)} labeled turns to {args.output}.")


if __name__ == "__main__":
    main()
