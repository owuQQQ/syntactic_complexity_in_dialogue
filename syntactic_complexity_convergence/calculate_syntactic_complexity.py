import argparse
from pathlib import Path

import stanza

from syntactic_complexity_convergence.calculate_metrics import calculate_with_pipeline


def main() -> None:
    """Parse command-line options, initialize Stanza, and calculate complexity."""
    parser = argparse.ArgumentParser(
        description="Use Stanza dependency parsing to calculate syntactic complexity per dialogue turn."
    )
    parser.add_argument("--input", required=True, type=Path, help="Canonical turn CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output metrics CSV.")
    parser.add_argument("--language", required=True, choices=["en", "de", "ja"])
    parser.add_argument("--gpu", action="store_true", help="Use Stanza's GPU parser if available.")
    args = parser.parse_args()

    # Stanza tokenizes and parses each utterance before dependency metrics are calculated.
    pipeline = stanza.Pipeline(
        args.language,
        processors="tokenize,pos,lemma,depparse",
        use_gpu=args.gpu,
        verbose=False,
    )
    calculate_with_pipeline(args.input, args.output, pipeline)


if __name__ == "__main__":
    main()
