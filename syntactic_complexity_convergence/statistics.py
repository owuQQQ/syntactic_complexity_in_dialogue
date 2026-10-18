import numpy as np


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
