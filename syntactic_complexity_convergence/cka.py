from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from syntactic_complexity_convergence.calculate_metrics import METRICS
from syntactic_complexity_convergence.statistics import benjamini_hochberg, phase_permutation_test


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA for two matrices with the same observations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be matrices with the same number of rows.")
    if x.shape[0] < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("CKA requires at least two finite observations.")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, "fro") * np.linalg.norm(yy, "fro")
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.clip(np.linalg.norm(cross, "fro") ** 2 / denominator, 0.0, 1.0))


def _resample(values: np.ndarray, size: int) -> np.ndarray:
    """Interpolate a multivariate trajectory to a fixed number of positions."""
    if len(values) == size:
        return values.astype(float, copy=True)
    old_position = np.linspace(0.0, 1.0, len(values))
    new_position = np.linspace(0.0, 1.0, size)
    return np.column_stack([
        np.interp(new_position, old_position, values[:, column])
        for column in range(values.shape[1])
    ])


def dialogue_phase_distances(
    frame: pd.DataFrame,
    metrics: list[str],
    n_phases: int,
    min_turns: int,
    points_per_phase: int,
    initiator: str,
    follower: str,
) -> np.ndarray:
    """Calculate per-dialogue linear CKA distances for each conversation phase."""
    standardized = frame.copy()
    means = standardized[metrics].mean()
    scales = standardized[metrics].std(ddof=0).replace(0, 1.0)
    standardized[metrics] = (standardized[metrics] - means) / scales
    dialogue_values: list[list[float]] = []
    for _, dialogue in standardized.groupby("dialogue_id", sort=False):
        sides = {
            role: group.sort_values("turn_index", kind="stable")
            for role, group in dialogue.groupby("speaker", sort=False)
        }
        if initiator not in sides or follower not in sides:
            continue
        if min(len(sides[initiator]), len(sides[follower])) < min_turns:
            continue
        values: list[float] = []
        valid = True
        for phase in range(n_phases):
            matrices = []
            for role in (initiator, follower):
                phase_column = f"phase_{n_phases}"
                if phase_column in sides[role]:
                    phase_values = sides[role].loc[
                        sides[role][phase_column] == phase + 1, metrics
                    ].to_numpy(dtype=float)
                else:
                    phases = np.array_split(
                        sides[role][metrics].to_numpy(dtype=float), n_phases
                    )
                    phase_values = phases[phase]
                if len(phase_values) < 2:
                    valid = False
                    break
                matrices.append(_resample(phase_values, points_per_phase))
            if not valid:
                break
            similarity = linear_cka(matrices[0], matrices[1])
            if not np.isfinite(similarity):
                valid = False
                break
            values.append(1.0 - similarity)
        if valid:
            dialogue_values.append(values)
    return np.asarray(dialogue_values, dtype=float).reshape(-1, n_phases)


def analyze(
    input_path: Path,
    output_prefix: Path,
    metrics: list[str] | None = None,
    n_phases: int = 8,
    min_turns: int = 16,
    points_per_phase: int = 20,
    permutations: int = 9999,
    seed: int = 42,
    initiator: str = "initiator",
    follower: str = "follower",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze CKA distance trends and write phase and permutation-test results."""
    if n_phases < 2 or min_turns < 2 * n_phases or points_per_phase < 2:
        raise ValueError(
            "Require phases >= 2, min_turns >= 2 * phases, and at least two CKA points."
        )
    selected = metrics or METRICS
    frame = pd.read_csv(input_path)
    required = {"dialogue_id", "speaker", "turn_index", *selected}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metrics input is missing columns: {sorted(missing)}")
    values = dialogue_phase_distances(
        frame, selected, n_phases, min_turns, points_per_phase, initiator, follower
    )
    if len(values) < 2:
        raise ValueError(
            f"Only {len(values)} valid paired dialogues; at least two are required."
        )
    observed, p_value, _ = phase_permutation_test(
        values, permutations=permutations, seed=seed
    )
    phases = values.mean(axis=0)
    phase_frame = pd.DataFrame({
        "phase": np.arange(1, n_phases + 1),
        "mean_cka_distance": phases,
        "n_dialogues": len(values),
    })
    summary = pd.DataFrame([{
        "n_dialogues": len(values),
        "n_phases": n_phases,
        "points_per_phase": points_per_phase,
        "slope": observed,
        "permutation_p_value": p_value,
        "directional_convergence": observed < 0 and p_value < 0.05,
    }])
    summary["permutation_p_value_bh"] = benjamini_hochberg(
        summary["permutation_p_value"].tolist()
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    phase_path = output_prefix.with_name(output_prefix.name + "_cka_phases.csv")
    summary_path = output_prefix.with_name(output_prefix.name + "_cka_test.csv")
    phase_frame.to_csv(phase_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote CKA phase distances to {phase_path} and trend test to {summary_path}.")
    return phase_frame, summary


def main() -> None:
    """Parse CLI arguments and run the linear CKA analysis."""
    parser = argparse.ArgumentParser(
        description="Measure paired-speaker linear CKA by dialogue phase and test its trend."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument("--phases", type=int, default=8)
    parser.add_argument("--min-turns", type=int, default=16)
    parser.add_argument("--points-per-phase", type=int, default=20)
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
        args.points_per_phase,
        args.permutations,
        args.seed,
        args.initiator,
        args.follower,
    )


if __name__ == "__main__":
    main()
