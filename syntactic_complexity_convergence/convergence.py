from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from syntactic_complexity_convergence.calculate_metrics import METRICS
from syntactic_complexity_convergence.statistics import benjamini_hochberg, phase_permutation_test


def paired_phase_gaps(
    frame: pd.DataFrame,
    metrics: list[str],
    n_phases: int,
    min_turns: int,
    initiator: str,
    follower: str,
) -> dict[str, np.ndarray]:
    """Compute per-dialogue phase gaps for the requested metrics."""
    gaps: dict[str, list[list[float]]] = {metric: [] for metric in metrics}
    for _, dialogue in frame.groupby("dialogue_id", sort=False):
        sides = {
            role: group.sort_values("turn_index", kind="stable")
            for role, group in dialogue.groupby("speaker", sort=False)
        }
        if initiator not in sides or follower not in sides:
            continue
        if min(len(sides[initiator]), len(sides[follower])) < min_turns:
            continue
        phase_metrics: dict[str, list[float]] = {metric: [] for metric in metrics}
        valid_dialogue = True
        for role in (initiator, follower):
            if f"phase_{n_phases}" in sides[role]:
                phase_ids = sides[role][f"phase_{n_phases}"].to_numpy()
                role_phases = [
                    sides[role].loc[phase_ids == phase, metrics].to_numpy(dtype=float)
                    for phase in range(1, n_phases + 1)
                ]
            else:
                role_phases = np.array_split(
                    sides[role][metrics].to_numpy(dtype=float), n_phases
                )
            for metric_index, metric in enumerate(metrics):
                medians = [
                    float(np.median(phase[:, metric_index])) if len(phase) else np.nan
                    for phase in role_phases
                ]
                if not np.isfinite(medians).all():
                    valid_dialogue = False
                phase_metrics[metric].append(medians)
        if valid_dialogue:
            for metric in metrics:
                left, right = phase_metrics[metric]
                gaps[metric].append(np.abs(np.asarray(left) - np.asarray(right)).tolist())
    return {
        metric: np.asarray(values, dtype=float).reshape(-1, n_phases)
        for metric, values in gaps.items()
    }


def analyze(
    input_path: Path,
    output_prefix: Path,
    metrics: list[str] | None = None,
    n_phases: int = 8,
    min_turns: int = 16,
    permutations: int = 9999,
    seed: int = 42,
    initiator: str = "initiator",
    follower: str = "follower",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test speaker-gap trends and write phase summaries and permutation results."""
    if n_phases < 2 or min_turns < n_phases or permutations < 1:
        raise ValueError("Require phases >= 2, min_turns >= phases, and permutations >= 1.")
    selected = metrics or METRICS
    frame = pd.read_csv(input_path)
    required = {"dialogue_id", "speaker", "turn_index", *selected}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metrics input is missing columns: {sorted(missing)}")
    dialogue_gaps = paired_phase_gaps(
        frame, selected, n_phases, min_turns, initiator, follower
    )
    phase_rows = []
    summary_rows = []
    for offset, metric in enumerate(selected):
        values = dialogue_gaps[metric]
        if len(values) < 2:
            raise ValueError(
                f"Metric {metric!r} has {len(values)} paired dialogues; "
                "at least two are needed for the phase permutation test."
            )
        observed, p_value, _ = phase_permutation_test(
            values, permutations=permutations, seed=seed + offset
        )
        means = values.mean(axis=0)
        for phase_index, mean_gap in enumerate(means, start=1):
            phase_rows.append({
                "metric": metric,
                "phase": phase_index,
                "mean_absolute_speaker_gap": mean_gap,
                "n_dialogues": len(values),
            })
        summary_rows.append({
            "metric": metric,
            "n_dialogues": len(values),
            "n_phases": n_phases,
            "slope": observed,
            "permutation_p_value": p_value,
            "directional_convergence": observed < 0 and p_value < 0.05,
        })
    summary = pd.DataFrame(summary_rows)
    summary["permutation_p_value_bh"] = benjamini_hochberg(
        summary["permutation_p_value"].tolist()
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    phase_path = output_prefix.with_name(output_prefix.name + "_phases.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_tests.csv")
    pd.DataFrame(phase_rows).to_csv(phase_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote phase gaps to {phase_path} and permutation tests to {summary_path}.")
    return pd.DataFrame(phase_rows), summary


def main() -> None:
    """Parse CLI arguments and run the syntactic convergence analysis."""
    parser = argparse.ArgumentParser(
        description="Test whether paired speakers' syntactic complexity gap declines over time."
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV produced by calculate_metrics.")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--phases", type=int, default=8)
    parser.add_argument("--min-turns", type=int, default=16, help="Minimum turns per speaker.")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initiator", default="initiator")
    parser.add_argument("--follower", default="follower")
    args = parser.parse_args()
    analyze(
        args.input,
        args.output_prefix,
        [name.strip() for name in args.metrics.split(",") if name.strip()],
        args.phases,
        args.min_turns,
        args.permutations,
        args.seed,
        args.initiator,
        args.follower,
    )


if __name__ == "__main__":
    main()
