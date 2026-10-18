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


def dependency_metrics(text: str, pipeline) -> dict[str, float]:
    """Calculate prototype syntactic-complexity metrics from a Stanza parse.

    ``syn_value`` is 0.5 * (whitespace-token count / number of distinct head IDs)
    + 0.5 * maximum dependency-tree depth. Head ID zero (the root) is included.
    Dependency length is the mean absolute token-to-head index distance.
    Complex dependency ratio is the share of non-root arcs longer than two.
    """
    document = pipeline(text)
    depths: list[int] = []
    dependency_distances: list[int] = []
    long_dependencies = 0
    dependency_count = 0
    head_ids: set[int] = set()
    parser_word_count = 0
    surface_token_count = len(text.split())

    for sentence in document.sentences:
        words = sentence.words
        parser_word_count += len(words)
        children: dict[int, list[int]] = {}
        roots: list[int] = []
        for word in words:
            head_id = int(word.head)
            word_id = int(word.id)
            head_ids.add(head_id)
            if head_id == 0:
                roots.append(word_id)
                continue
            children.setdefault(head_id, []).append(word_id)
            distance = abs(word_id - head_id)
            dependency_distances.append(distance)
            dependency_count += 1
            # The prototype marks dependencies spanning more than two tokens.
            long_dependencies += distance > 2

        def depth(node_id: int) -> int:
            """Return the maximum dependency-tree depth below one node."""
            return 1 + max((depth(child) for child in children.get(node_id, [])), default=0)

        depths.extend(depth(root) for root in roots)

    tree_depth = max(depths, default=0)
    dependency_length = float(np.mean(dependency_distances)) if dependency_distances else 0.0
    complex_ratio = long_dependencies / dependency_count if dependency_count else 0.0
    # Keep the original prototype's equal weighting of normalized length and depth.
    syn_value = (
        0.5 * (surface_token_count / len(head_ids)) + 0.5 * tree_depth
        if head_ids
        else 0.5 * tree_depth
    )
    return {
        "word_count": parser_word_count,
        "syn_value": float(syn_value),
        "dependency_length": dependency_length,
        "tree_depth": float(tree_depth),
        "complex_dependency_ratio": float(complex_ratio),
    }


def calculate(input_path: Path, output_path: Path, language: str, use_gpu: bool = False) -> pd.DataFrame:
    """Parse every canonical turn and write its dependency-based metrics."""
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError(
            "Stanza is required for parsing. Install the subproject requirements, "
            "then download the selected language model (see README.md)."
        ) from exc

    turns = pd.read_csv(input_path)
    required = {"dialogue_id", "speaker", "turn_index", "text"}
    missing = required - set(turns.columns)
    if missing:
        raise ValueError(f"Canonical input is missing columns: {sorted(missing)}")
    pipeline = stanza.Pipeline(
        language,
        processors="tokenize,pos,lemma,depparse",
        use_gpu=use_gpu,
        verbose=False,
    )
    rows = []
    for row in turns.itertuples(index=False):
        metrics = dependency_metrics(str(row.text), pipeline)
        rows.append(metrics)
    output = pd.concat([turns.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Calculated metrics for {len(output)} turns; wrote {output_path}.")
    return output


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
