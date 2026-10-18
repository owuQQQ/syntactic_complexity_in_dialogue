import numpy as np

from joblib import Parallel, delayed


def slope(values: np.ndarray) -> float:
    """Fit and return the ordinary least-squares slope across ordered phases."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("A slope requires at least two phase values.")
    x = np.arange(1, len(values) + 1, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def phase_permutation_test(
    phase_values: np.ndarray,
    permutations: int = 9999,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Test a negative phase trend against independent within-dialogue shuffles."""
    values = np.asarray(phase_values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("Expected at least two dialogues and two phases.")
    if not np.isfinite(values).all():
        raise ValueError("Phase values must be finite.")
    if permutations < 1:
        raise ValueError("At least one permutation is required.")
    observed = slope(values.mean(axis=0))
    rng = np.random.default_rng(seed)
    null_slopes = np.empty(permutations, dtype=float)
    for iteration in range(permutations):
        shuffled = np.vstack([rng.permutation(row) for row in values])
        null_slopes[iteration] = slope(shuffled.mean(axis=0))
    p_value = (1 + np.count_nonzero(null_slopes <= observed)) / (permutations + 1)
    return observed, float(p_value), null_slopes


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in their original order."""
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite values between 0 and 1.")
    order = np.argsort(p)
    adjusted_sorted = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def build_groups_from_df(
    frame,
    metrics: list[str],
    keep_order: bool = True,
    id_column: str | None = None,
) -> list[dict]:
    """Convert dialogue rows to the grouped-array structure used by the legacy test."""
    if id_column is None:
        id_column = "dialogue_id" if "dialogue_id" in frame else "Index"
    required = {id_column, *metrics}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    source = frame
    if not keep_order and "turn_index" in source:
        source = source.sort_values("turn_index", kind="stable")
    groups = []
    for dialogue_id, dialogue in source.groupby(id_column, sort=False):
        values = {
            metric: dialogue[metric].to_numpy(dtype=float)
            for metric in metrics
        }
        groups.append({
            "id": dialogue_id,
            "len": len(dialogue),
            "metrics": values,
        })
    return groups


def _phase_gap_sequences(
    groups1: list[dict],
    groups2: list[dict],
    metrics: list[str],
    n_phases: int,
) -> dict[str, np.ndarray]:
    """Calculate paired per-dialogue phase-median gaps for grouped turn data."""
    second_by_id = {group["id"]: group for group in groups2}
    sequences = {metric: [] for metric in metrics}
    for first in groups1:
        second = second_by_id.get(first["id"])
        if second is None:
            continue
        if min(first["len"], second["len"]) < n_phases:
            continue
        for metric in metrics:
            first_phases = np.array_split(first["metrics"][metric], n_phases)
            second_phases = np.array_split(second["metrics"][metric], n_phases)
            if any(not len(phase) for phase in first_phases + second_phases):
                continue
            first_medians = np.asarray(
                [np.nanmedian(phase) for phase in first_phases],
                dtype=float,
            )
            second_medians = np.asarray(
                [np.nanmedian(phase) for phase in second_phases],
                dtype=float,
            )
            if np.isfinite(first_medians).all() and np.isfinite(second_medians).all():
                sequences[metric].append(np.abs(first_medians - second_medians))
    return {
        metric: np.asarray(values, dtype=float).reshape(-1, n_phases)
        for metric, values in sequences.items()
    }


def _decreasing_ratios(
    sequences: dict[str, np.ndarray],
    min_decreases: int = 2,
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate the legacy loose and strict per-dialogue decreasing ratios."""
    loose = {}
    strict = {}
    for metric, values in sequences.items():
        if not len(values):
            loose[metric] = 0.0
            strict[metric] = 0.0
            continue
        endpoint_decreases = values[:, 0] > values[:, -1]
        loose[metric] = float(np.mean(endpoint_decreases))
        adjacent_decreases = np.count_nonzero(np.diff(values, axis=1) < 0, axis=1)
        strict[metric] = float(np.mean(
            endpoint_decreases & (adjacent_decreases >= min_decreases)
        ))
    return strict, loose


def strict_decreasing_from_groups(
    groups1: list[dict],
    groups2: list[dict],
    metrics: list[str],
    n_phases: int,
) -> tuple[dict[str, float], dict[str, list[list[float]]]]:
    """Return strict convergence ratios and paired phase gaps, as in the old analysis."""
    sequences = _phase_gap_sequences(groups1, groups2, metrics, n_phases)
    strict, _ = _decreasing_ratios(sequences)
    return strict, {
        metric: values.tolist()
        for metric, values in sequences.items()
    }


def one_permutation(
    groups1: list[dict],
    groups2: list[dict],
    metrics: list[str],
    n_phases: int,
    rng_seed: int | None = None,
) -> dict[str, float]:
    """Shuffle each speaker's turn order independently and recalculate strict ratios."""
    rng = np.random.default_rng(rng_seed)

    def shuffled(groups: list[dict]) -> list[dict]:
        """Shuffle the same turn permutation across metrics within each dialogue."""
        result = []
        for group in groups:
            indices = rng.permutation(group["len"])
            result.append({
                "id": group["id"],
                "len": group["len"],
                "metrics": {
                    metric: group["metrics"][metric][indices]
                    for metric in metrics
                },
            })
        return result

    permuted1 = shuffled(groups1)
    permuted2 = shuffled(groups2)
    return strict_decreasing_from_groups(
        permuted1, permuted2, metrics, n_phases
    )[0]


def choose_B_auto(n_dialogues: int, B_user: int | None = None) -> int:
    """Choose the legacy permutation count for a corpus unless explicitly supplied."""
    if B_user is not None:
        if B_user < 2:
            raise ValueError("At least two permutations are required.")
        return B_user
    if n_dialogues < 50:
        return max(300, n_dialogues * 15)
    if n_dialogues < 500:
        return 750
    if n_dialogues < 1000:
        return 1500
    return 1750


def permutation_test_fast(
    groups1: list[dict],
    groups2: list[dict],
    metrics: list[str],
    n_phases: int,
    B_user: int | None = None,
    n_jobs: int = 1,
    random_state: int = 42,
    batch_size: int = 100,
) -> dict[str, object]:
    """Build a null distribution by independently permuting paired turn orders."""
    if n_jobs == 0 or batch_size < 1:
        raise ValueError("n_jobs cannot be zero and batch_size must be positive.")
    first_ids = {group["id"] for group in groups1}
    second_ids = {group["id"] for group in groups2}
    n_dialogues = len(first_ids & second_ids)
    permutations = choose_B_auto(n_dialogues, B_user)
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**31 - 1, size=permutations)
    null_values = {metric: [] for metric in metrics}

    for start in range(0, permutations, batch_size):
        batch_seeds = seeds[start:start + batch_size]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(one_permutation)(
                groups1, groups2, metrics, n_phases, int(seed)
            )
            for seed in batch_seeds
        )
        for result in batch_results:
            for metric in metrics:
                null_values[metric].append(result[metric])

    null_mean = {
        metric: float(np.mean(null_values[metric]))
        for metric in metrics
    }
    null_std = {
        metric: float(np.std(null_values[metric], ddof=1))
        for metric in metrics
    }
    return {
        "null_vals": null_values,
        "null_mean": null_mean,
        "null_std": null_std,
        "B_used": permutations,
    }


def cohen_d_from_null(
    observed: dict[str, float],
    null_mean: dict[str, float],
    null_std: dict[str, float],
) -> dict[str, float]:
    """Standardize observed convergence ratios against their permutation null."""
    return {
        metric: (
            float((value - null_mean[metric]) / null_std[metric])
            if null_std[metric] > 0
            else float("nan")
        )
        for metric, value in observed.items()
    }


def bootstrap_ci_diff(
    sequences: dict[str, np.ndarray],
    metrics: list[str],
    null_mean: dict[str, float],
    n_boot: int = 1000,
    random_state: int = 123,
) -> tuple[dict[str, tuple[float, float]], dict[str, list[float]]]:
    """Bootstrap dialogue-level strict-ratio differences from the permutation null."""
    if n_boot < 1:
        raise ValueError("At least one bootstrap sample is required.")
    rng = np.random.default_rng(random_state)
    differences = {metric: [] for metric in metrics}
    for metric in metrics:
        values = sequences[metric]
        if not len(values):
            differences[metric] = [float("nan")] * n_boot
            continue
        for _ in range(n_boot):
            indices = rng.integers(0, len(values), size=len(values))
            strict, _ = _decreasing_ratios({metric: values[indices]})
            differences[metric].append(strict[metric] - null_mean[metric])
    intervals = {
        metric: (
            float(np.percentile(differences[metric], 2.5)),
            float(np.percentile(differences[metric], 97.5)),
        )
        if np.isfinite(differences[metric]).all()
        else (float("nan"), float("nan"))
        for metric in metrics
    }
    return intervals, differences


def strict_convergence_test(
    groups1: list[dict],
    groups2: list[dict],
    metrics: list[str],
    n_phases: int,
    min_turns: int,
    permutations: int = 9999,
    bootstrap_samples: int = 1000,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict[str, dict[str, float]]:
    """Run the legacy strict-ratio permutation and dialogue-bootstrap analysis."""
    second_by_id = {group["id"]: group for group in groups2}
    valid1 = []
    valid2 = []
    for group in groups1:
        partner = second_by_id.get(group["id"])
        if partner is not None and min(group["len"], partner["len"]) >= min_turns:
            valid1.append(group)
            valid2.append(partner)
    if len(valid1) < 2:
        raise ValueError(
            f"Only {len(valid1)} paired dialogues meet the {min_turns}-turn minimum; "
            "at least two are required."
        )

    observed, _ = strict_decreasing_from_groups(valid1, valid2, metrics, n_phases)
    sequences = _phase_gap_sequences(valid1, valid2, metrics, n_phases)
    _, loose = _decreasing_ratios(sequences)
    null = permutation_test_fast(
        valid1,
        valid2,
        metrics,
        n_phases,
        B_user=permutations,
        random_state=seed,
        n_jobs=n_jobs,
    )
    intervals, _ = bootstrap_ci_diff(
        sequences,
        metrics,
        null["null_mean"],
        n_boot=bootstrap_samples,
        random_state=seed + 1,
    )
    effects = cohen_d_from_null(observed, null["null_mean"], null["null_std"])

    results = {}
    for metric in metrics:
        null_values = np.asarray(null["null_vals"][metric], dtype=float)
        p_value = (
            1 + np.count_nonzero(null_values >= observed[metric])
        ) / (len(null_values) + 1)
        results[metric] = {
            "strict_decreasing_ratio": observed[metric],
            "loose_decreasing_ratio": loose[metric],
            "strict_null_mean": null["null_mean"][metric],
            "strict_null_std": null["null_std"][metric],
            "strict_permutation_p_value": float(p_value),
            "strict_cohen_d": effects[metric],
            "strict_ci_lower": intervals[metric][0],
            "strict_ci_upper": intervals[metric][1],
        }
    return results
