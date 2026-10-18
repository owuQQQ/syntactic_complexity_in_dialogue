#!/usr/bin/env python3
"""Extract poem text from all JSON files in a folder."""

import argparse
import json
import os
import re
from glob import glob


def _num_key(label: str) -> int:
    match = re.search(r"(\d+)", label)
    return int(match.group(1)) if match else 0


def extract_text(data: dict) -> str:
    if not isinstance(data, dict) or not data:
        return ""

    root = next(iter(data.values()))
    poem = root.get("poem", {}) if isinstance(root, dict) else {}

    lines = []
    for stanza_key in sorted(poem.keys(), key=_num_key):
        stanza = poem.get(stanza_key, {})
        if not isinstance(stanza, dict):
            continue
        for line_key in sorted(stanza.keys(), key=_num_key):
            line = stanza.get(line_key, {})
            if isinstance(line, dict) and "text" in line:
                lines.append(line["text"])
            elif isinstance(line, str):
                lines.append(line)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract poem text from JSON files.")
    parser.add_argument("input_dir", help="Folder containing JSON files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder to write .txt outputs (defaults to input_dir)",
    )
    parser.add_argument(
        "--combine",
        default=None,
        help="Optional path to write a single combined text file",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir or input_dir)
    os.makedirs(output_dir, exist_ok=True)

    json_paths = sorted(glob(os.path.join(input_dir, "*.json")))
    if not json_paths:
        raise SystemExit(f"No .json files found in {input_dir}")

    combined_parts = []

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = extract_text(data)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{base}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        if args.combine:
            combined_parts.append(f"### {base}\n{text}\n")

    if args.combine:
        combine_path = os.path.abspath(args.combine)
        with open(combine_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_parts).rstrip() + "\n")


if __name__ == "__main__":
    main()
