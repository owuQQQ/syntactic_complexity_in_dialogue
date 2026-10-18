import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    "syn_value",
    "dependency_length",
    "tree_depth",
    "complex_dependency_ratio",
]


def _tree_depth_from_document(document) -> int:
    """Compute maximum dependency depth in the first sentence, as in the original script."""
    if not document.sentences:
        return 0
    sentence = document.sentences[0]
    children: dict[int, list[int]] = {}
    roots = []
    for word in sentence.words:
        if word.head == 0:
            roots.append(word.id)
        else:
            children.setdefault(word.head, []).append(word.id)

    def compute_depth(node_id: int) -> int:
        """Return one node's depth by recursively traversing its dependents."""
        return 1 + max(
            (compute_depth(child) for child in children.get(node_id, [])),
            default=0,
        )

    return max((compute_depth(root) for root in roots), default=0)


def dependency_metrics(text: str, pipeline) -> dict[str, float]:
    """Calculate the original project's four syntactic-complexity metrics.

    ``syn_value`` is 0.5 * (whitespace-token count / number of distinct head IDs)
    + 0.5 * first-sentence tree depth. Root head ID zero is included, matching
    ``calculate_complexity`` in the earlier project script.
    """
    document = pipeline(text)
    head_ids = {
        word.head
        for sentence in document.sentences
        for word in sentence.words
    }
    distances = [
        abs(word.id - word.head)
        for sentence in document.sentences
        for word in sentence.words
        if word.head > 0
    ]
    long_dependencies = sum(distance > 2 for distance in distances)
    tree = _tree_depth_from_document(document)
    complexity = (
        0.5 * (len(text.split()) / len(head_ids)) + 0.5 * tree
        if head_ids
        else 0.5 * tree
    )
    return {
        "word_count": sum(len(sentence.words) for sentence in document.sentences),
        "syn_value": float(complexity),
        "dependency_length": float(np.mean(distances)) if distances else 0.0,
        "tree_depth": float(tree),
        "complex_dependency_ratio": (
            long_dependencies / len(distances) if distances else 0.0
        ),
    }


def tree_depth(text: str, dep_parser) -> int:
    """Parse one utterance and return its first-sentence dependency-tree depth."""
    return _tree_depth_from_document(dep_parser(text))


def dependency_length(text: str, dep_parser) -> float:
    """Return the mean token-to-head distance across non-root dependency arcs."""
    document = dep_parser(text)
    distances = [
        abs(word.id - word.head)
        for sentence in document.sentences
        for word in sentence.words
        if word.head > 0
    ]
    return float(np.mean(distances)) if distances else 0.0


def calculate_complexity(sentence: str, dep_parser) -> float:
    """Return the original weighted length-and-tree-depth complexity score."""
    return dependency_metrics(sentence, dep_parser)["syn_value"]


def long_dependency_ratio(text: str, dep_parser, threshold: int = 3) -> float:
    """Return the proportion of non-root dependencies longer than a threshold."""
    document = dep_parser(text)
    distances = [
        abs(word.id - word.head)
        for sentence in document.sentences
        for word in sentence.words
        if word.head > 0
    ]
    return (
        sum(distance > threshold for distance in distances) / len(distances)
        if distances
        else 0.0
    )


def sentence_syncom_metrics(text: str, dep_parser) -> list[float]:
    """Return complexity, dependency length, tree depth, and long-arc ratio."""
    metrics = dependency_metrics(text, dep_parser)
    return [metrics[name] for name in METRICS]


def calculate_syncom(texts, dep_parser) -> np.ndarray:
    """Calculate the four syntactic metrics for each non-empty text."""
    rows = [
        sentence_syncom_metrics(str(text).strip(), dep_parser)
        for text in texts
        if str(text).strip()
    ]
    return np.asarray(rows, dtype=float) if rows else np.empty((0, len(METRICS)))


def summarize_syncom(sentences, dep_parser):
    """Return the mean metrics for a sentence collection, or ``None`` if empty."""
    metrics = calculate_syncom(sentences, dep_parser)
    return metrics.mean(axis=0) if metrics.size else None


def calculate_with_pipeline(
    input_path: Path,
    output_path: Path,
    pipeline,
) -> pd.DataFrame:
    """Apply an initialized Stanza pipeline and save metrics for every dialogue turn."""
    turns = pd.read_csv(input_path)
    required = {"dialogue_id", "speaker", "turn_index", "text"}
    missing = required - set(turns.columns)
    if missing:
        raise ValueError(f"Canonical input is missing columns: {sorted(missing)}")
    rows = []
    for row in turns.itertuples(index=False):
        metrics = dependency_metrics(str(row.text), pipeline)
        rows.append(metrics)
    output = pd.concat([turns.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Calculated metrics for {len(output)} turns; wrote {output_path}.")
    return output


def calculate(input_path: Path, output_path: Path, language: str, use_gpu: bool = False) -> pd.DataFrame:
    """Initialize Stanza for a language and calculate metrics for canonical turns."""
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError(
            "Stanza is required for parsing. Install the subproject requirements, "
            "then download the selected language model (see README.md)."
        ) from exc
    pipeline = stanza.Pipeline(
        language,
        processors="tokenize,pos,lemma,depparse",
        use_gpu=use_gpu,
        verbose=False,
    )
    return calculate_with_pipeline(input_path, output_path, pipeline)


def main() -> None:
    """Parse CLI arguments and calculate metrics for a canonical turn CSV."""
    parser = argparse.ArgumentParser(description="Calculate dependency-based complexity per turn.")
    parser.add_argument("--input", required=True, type=Path, help="Canonical turn CSV.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", required=True, choices=["en", "de", "ja"])
    parser.add_argument("--gpu", action="store_true", help="Use Stanza's GPU parser if available.")
    args = parser.parse_args()
    calculate(args.input, args.output, args.language, args.gpu)


if __name__ == "__main__":
    main()
