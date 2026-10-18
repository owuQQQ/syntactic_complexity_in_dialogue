from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ("sid", "dialogue_id", "id", "Index")


def _read_id_csv(path: Path) -> pd.DataFrame:
    """Read a CSV while preserving identifiers that may contain leading zeros."""
    return pd.read_csv(path, dtype={column: str for column in ID_COLUMNS})


def load_spectrum(path: Path) -> dict[str, np.ndarray]:
    """Load a long-form Fourier spectrum and group power values by dialogue."""
    frame = _read_id_csv(path)
    id_column = next((name for name in ID_COLUMNS if name in frame), None)
    if id_column is None or "power" not in frame:
        raise ValueError(f"{path} must be a long CSV with an id column and a 'power' column.")
    groups: dict[str, np.ndarray] = {}
    for dialogue_id, group in frame.groupby(id_column, sort=False):
        if "freq" in group:
            group = group.sort_values("freq", kind="stable")
        values = pd.to_numeric(group["power"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) >= 2 and np.isfinite(values).all():
            groups[str(dialogue_id)] = values
    if not groups:
        raise ValueError(f"No valid per-dialogue spectra found in {path}.")
    return groups


def load_lengths(path: Path) -> dict[str, float]:
    """Load one nonnegative dialogue-length value per dialogue ID."""
    frame = _read_id_csv(path)
    id_column = next((name for name in ID_COLUMNS if name in frame), None)
    length_column = next(
        (name for name in ("length", "word_count", "token_count", "turn_count") if name in frame),
        None,
    )
    if id_column is None or length_column is None:
        raise ValueError(f"{path} must contain an id column and length/word_count/token_count.")
    frame[length_column] = pd.to_numeric(frame[length_column], errors="coerce")
    frame = frame.dropna(subset=[length_column])
    if not np.isfinite(frame[length_column]).all() or (frame[length_column] < 0).any():
        raise ValueError(f"{path} contains negative or non-finite dialogue lengths.")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate dialogue IDs.")
    return {
        str(row[id_column]): float(row[length_column])
        for _, row in frame.iterrows()
    }


def _load_dialogue_series(path: Path, value_name: str) -> dict[str, np.ndarray]:
    """Read finite, whitespace-separated values keyed by tab-separated dialogue IDs."""
    series: dict[str, np.ndarray] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{line_number} must contain an ID and a tab-separated {value_name} series."
                )
            try:
                values = np.asarray([float(value) for value in parts[1].split()], dtype=float)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} contains a nonnumeric {value_name} value."
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{path}:{line_number} contains a non-finite {value_name} value."
                )
            if len(values) >= 2:
                if parts[0] in series:
                    raise ValueError(f"{path}:{line_number} repeats dialogue ID {parts[0]!r}.")
                series[parts[0]] = values
    if not series:
        raise ValueError(f"No valid per-dialogue {value_name} series found in {path}.")
    return series


def load_surprisal_series(path: Path) -> dict[str, np.ndarray]:
    """Read token-level surprisal values from an ID-tab-series text file."""
    return _load_dialogue_series(path, "surprisal")


def load_nll_series(path: Path) -> dict[str, np.ndarray]:
    """Read per-token negative log-likelihood values keyed by dialogue ID."""
    return _load_dialogue_series(path, "NLL")


def load_semcom_surprisal(path: Path) -> dict[str, np.ndarray]:
    """Extract and concatenate utterance surprisal values from semcom text output."""
    series: dict[str, list[float]] = {}
    dialogue_id = None
    metric = None
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("index:"):
                dialogue_id = line.split(":", maxsplit=1)[1].strip()
                if not dialogue_id:
                    raise ValueError(f"{path}:{line_number} has an empty dialogue ID.")
                series.setdefault(dialogue_id, [])
                metric = None
                continue
            if line.lower() in {"surprisal", "shannon", "renyi"}:
                metric = line.lower()
                continue
            if metric != "surprisal":
                continue
            if dialogue_id is None:
                raise ValueError(f"{path}:{line_number} has surprisal values before an index header.")
            try:
                values = [float(value) for value in line.split()]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number} contains a nonnumeric surprisal value."
                ) from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line_number} contains a non-finite surprisal value.")
            series[dialogue_id].extend(values)
    result = {
        dialogue_id: np.asarray(values, dtype=float)
        for dialogue_id, values in series.items()
        if len(values) >= 2
    }
    if not result:
        raise ValueError(f"No valid dialogue-level surprisal series found in {path}.")
    return result


def series_to_spectra(series: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute FourierGPT-compatible one-sided magnitude spectra for each dialogue."""
    return {
        dialogue_id: np.abs(np.fft.rfft(values))
        for dialogue_id, values in series.items()
    }


def _fixed_width(values: np.ndarray, width: int) -> np.ndarray:
    """Interpolate a one-dimensional feature series to a fixed width."""
    if width < 2:
        raise ValueError("Feature width must be at least 2.")
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, width)
    return np.interp(target, source, values)


def _length_match(
    records: list[tuple[str, int, float, np.ndarray]],
    bins: int,
    seed: int,
) -> list[tuple[str, int, float, np.ndarray]]:
    """Balance labels within pooled dialogue-length quantile bins."""
    if bins < 1:
        raise ValueError("Length-match bins must be positive.")
    lengths = np.asarray([row[2] for row in records], dtype=float)
    edges = np.quantile(lengths, np.linspace(0, 1, bins + 1))
    assignments = np.digitize(lengths, edges[1:-1], right=True)
    rng = np.random.default_rng(seed)
    selected = []
    for bin_index in range(bins):
        in_bin = np.flatnonzero(assignments == bin_index)
        human = in_bin[[records[i][1] == 0 for i in in_bin]]
        llm = in_bin[[records[i][1] == 1 for i in in_bin]]
        count = min(len(human), len(llm))
        if count:
            selected.extend(rng.choice(human, count, replace=False).tolist())
            selected.extend(rng.choice(llm, count, replace=False).tolist())
    return [records[index] for index in sorted(selected)]


def _scores(
    feature_sets: tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
    lengths: tuple[dict[str, float], dict[str, float]],
    width: int,
    bins: int,
    seed: int,
    test_size: float,
    method: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Score one feature family and its length-only baseline on matched data."""
    records: list[tuple[str, int, float, np.ndarray]] = []
    for label, (samples, length_map) in enumerate(zip(feature_sets, lengths)):
        missing_lengths = set(samples) - set(length_map)
        if missing_lengths:
            raise ValueError(
                f"{len(missing_lengths)} dialogue IDs in the {('human', 'LLM')[label]} {method} "
                "input have no matching length. Supply a complete length table."
            )
        for dialogue_id, values in samples.items():
            records.append((
                dialogue_id,
                label,
                float(length_map[dialogue_id]),
                _fixed_width(values, width),
            ))
    records = _length_match(records, bins, seed)
    counts = [sum(row[1] == label for row in records) for label in (0, 1)]
    if min(counts) < 4:
        raise ValueError(
            f"Length matching retained only {counts[0]} human and {counts[1]} LLM samples; "
            "at least four per class are needed. Reduce --length-bins or check length overlap."
        )
    y = np.asarray([row[1] for row in records], dtype=int)
    x = np.vstack([row[3] for row in records])
    length_values = np.asarray([row[2] for row in records], dtype=float).reshape(-1, 1)
    train_ids, test_ids = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    def evaluate(features: np.ndarray, model_name: str) -> dict[str, object]:
        """Fit a logistic model and return metrics on the held-out split."""
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        model.fit(features[train_ids], y[train_ids])
        prediction = model.predict(features[test_ids])
        probability = model.predict_proba(features[test_ids])[:, 1]
        return {
            "method": method,
            "model": model_name,
            "n_human": counts[0],
            "n_llm": counts[1],
            "n_test": len(test_ids),
            "accuracy": accuracy_score(y[test_ids], prediction),
            "precision": precision_score(y[test_ids], prediction, zero_division=0),
            "recall": recall_score(y[test_ids], prediction, zero_division=0),
            "f1": f1_score(y[test_ids], prediction, zero_division=0),
            "roc_auc": roc_auc_score(y[test_ids], probability),
        }

    feature_score = evaluate(x, method)
    length_score = evaluate(np.log1p(length_values), "length_only_baseline")
    return feature_score, length_score


def analyze(
    output_path: Path,
    human_fft: Path | None = None,
    llm_fft: Path | None = None,
    human_lengths: Path | None = None,
    llm_lengths: Path | None = None,
    human_surprisal: Path | None = None,
    llm_surprisal: Path | None = None,
    human_nll: Path | None = None,
    llm_nll: Path | None = None,
    human_semcom: Path | None = None,
    llm_semcom: Path | None = None,
    width: int = 128,
    length_bins: int = 10,
    test_size: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare Fourier and/or surprisal features with length-only controls."""
    if not 0 < test_size < 1:
        raise ValueError("Test size must be between zero and one.")
    if (human_fft is None) != (llm_fft is None):
        raise ValueError("Provide both --human-fft and --llm-fft, or neither.")
    if (human_surprisal is None) != (llm_surprisal is None):
        raise ValueError("Provide both --human-surprisal and --llm-surprisal, or neither.")
    if (human_nll is None) != (llm_nll is None):
        raise ValueError("Provide both --human-nll and --llm-nll, or neither.")
    if (human_semcom is None) != (llm_semcom is None):
        raise ValueError("Provide both --human-semcom and --llm-semcom, or neither.")
    if human_surprisal is not None and human_semcom is not None:
        raise ValueError("Use either ID-tab-series surprisal or semcom inputs, not both.")
    if human_fft is not None and human_nll is not None:
        raise ValueError("Use either existing Fourier CSVs or raw NLL series, not both.")
    if human_fft is None and human_surprisal is None and human_nll is None and human_semcom is None:
        raise ValueError("At least one feature pair (Fourier, NLL, or surprisal) is required.")
    results = []
    if human_fft is not None or human_nll is not None:
        if human_lengths is None or llm_lengths is None:
            raise ValueError("Fourier spectra require --human-lengths and --llm-lengths for control.")
        if human_nll is not None:
            fft_features = (
                series_to_spectra(load_nll_series(human_nll)),
                series_to_spectra(load_nll_series(llm_nll)),
            )
        else:
            fft_features = (load_spectrum(human_fft), load_spectrum(llm_fft))
        scores = _scores(
            fft_features,
            (load_lengths(human_lengths), load_lengths(llm_lengths)),
            width,
            length_bins,
            seed,
            test_size,
            "fouriergpt_nll" if human_nll is not None else "fouriergpt",
        )
        results.extend(scores)
    if human_surprisal is not None:
        human_series = load_surprisal_series(human_surprisal)
        llm_series = load_surprisal_series(llm_surprisal)
        scores = _scores(
            (human_series, llm_series),
            (
                {key: float(len(values)) for key, values in human_series.items()},
                {key: float(len(values)) for key, values in llm_series.items()},
            ),
            width,
            length_bins,
            seed,
            test_size,
            "surprisal_baseline",
        )
        results.extend(scores)
    if human_semcom is not None:
        human_series = load_semcom_surprisal(human_semcom)
        llm_series = load_semcom_surprisal(llm_semcom)
        scores = _scores(
            (human_series, llm_series),
            (
                {key: float(len(values)) for key, values in human_series.items()},
                {key: float(len(values)) for key, values in llm_series.items()},
            ),
            width,
            length_bins,
            seed,
            test_size,
            "surprisal_baseline",
        )
        results.extend(scores)
    output = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote length-matched detection scores to {output_path}.")
    return output


def main() -> None:
    """Parse CLI arguments and run the requested detector comparisons."""
    parser = argparse.ArgumentParser(
        description="Compare FourierGPT and surprisal representations with length-only controls."
    )
    parser.add_argument("--human-fft", type=Path)
    parser.add_argument("--llm-fft", type=Path)
    parser.add_argument("--human-lengths", type=Path)
    parser.add_argument("--llm-lengths", type=Path)
    parser.add_argument("--human-surprisal", type=Path)
    parser.add_argument("--llm-surprisal", type=Path)
    parser.add_argument(
        "--human-nll",
        type=Path,
        help="ID-tab-series token NLL input; the script computes its Fourier spectrum.",
    )
    parser.add_argument("--llm-nll", type=Path)
    parser.add_argument(
        "--human-semcom",
        type=Path,
        help="Semcom result text; extracts token surprisal for the baseline.",
    )
    parser.add_argument("--llm-semcom", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--length-bins", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analyze(
        output_path=args.output,
        human_fft=args.human_fft,
        llm_fft=args.llm_fft,
        human_lengths=args.human_lengths,
        llm_lengths=args.llm_lengths,
        human_surprisal=args.human_surprisal,
        llm_surprisal=args.llm_surprisal,
        human_nll=args.human_nll,
        llm_nll=args.llm_nll,
        human_semcom=args.human_semcom,
        llm_semcom=args.llm_semcom,
        width=args.width,
        length_bins=args.length_bins,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
