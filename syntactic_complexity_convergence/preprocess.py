import argparse
import json
from pathlib import Path

import pandas as pd


CANONICAL_COLUMNS = ["dialogue_id", "speaker", "turn_index", "text"]
COLUMN_ALIASES = {
    "dialogue_id": ["dialogue_id", "Dialogue_id", "Dialogue_ID", "id", "Index"],
    "speaker": ["speaker", "Speaker", "role"],
    "turn_index": ["turn_index", "turn_id", "turn", "position"],
    "text": ["text", "utterance", "Utterance", "sentence", "Dialogue"],
}


def _records_from_json(payload, role: str | None) -> pd.DataFrame:
    """Convert supported JSON dialogue structures into a tabular frame."""
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for dialogue_id, turns in payload.items():
            if not isinstance(turns, list):
                raise ValueError(f"Dialogue {dialogue_id!r} must contain a list of turns.")
            for position, turn in enumerate(turns):
                if isinstance(turn, str):
                    rows.append({
                        "dialogue_id": dialogue_id,
                        "speaker": role,
                        "turn_index": position,
                        "text": turn,
                    })
                elif isinstance(turn, dict):
                    rows.append({
                        "dialogue_id": turn.get("dialogue_id", dialogue_id),
                        "speaker": turn.get("speaker", role),
                        "turn_index": turn.get("turn_index", position),
                        "text": turn.get("text", turn.get("utterance")),
                    })
                else:
                    raise ValueError(
                        f"Turn {position} in dialogue {dialogue_id!r} must be text or an object."
                    )
    else:
        raise ValueError("JSON input must be a list of turn records or a dialogue-to-turns object.")
    return pd.DataFrame(rows)


def _rename_columns(
    frame: pd.DataFrame,
    overrides: dict[str, str | None],
) -> pd.DataFrame:
    """Map common corpus-specific headers and optional overrides to canonical names."""
    rename = {}
    for target, candidates in COLUMN_ALIASES.items():
        chosen = overrides[target] or next(
            (candidate for candidate in candidates if candidate in frame.columns),
            None,
        )
        if chosen is not None and chosen in frame.columns:
            rename[chosen] = target
    return frame.rename(columns=rename)


def preprocess(
    input_path: Path,
    output_path: Path,
    file_format: str = "auto",
    id_column: str | None = None,
    speaker_column: str | None = None,
    turn_column: str | None = None,
    text_column: str | None = None,
    role: str | None = None,
) -> pd.DataFrame:
    """Convert CSV/JSON dialogue data to ordered, canonical turn records."""
    suffix_format = input_path.suffix.lower().lstrip(".")
    selected_format = suffix_format if file_format == "auto" else file_format
    overrides = {
        "dialogue_id": id_column,
        "speaker": speaker_column,
        "turn_index": turn_column,
        "text": text_column,
    }
    if selected_format == "json":
        raw = _records_from_json(
            json.loads(input_path.read_text(encoding="utf-8")),
            role=role,
        )
        raw = _rename_columns(raw, overrides)
    elif selected_format == "csv":
        raw = _rename_columns(pd.read_csv(input_path), overrides)
    else:
        raise ValueError("Input format must be csv, json, or auto.")

    required = {"dialogue_id", "text"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns {sorted(missing)}. "
            "Specify the source columns with the corresponding --*-column option."
        )
    if "speaker" not in raw.columns:
        if role is None:
            raise ValueError(
                "No speaker column was found. Specify --speaker-column, or --role "
                "when the file contains turns from just one speaker."
            )
        raw["speaker"] = role
    if role is not None:
        raw["speaker"] = role
    if "turn_index" not in raw.columns:
        raw["turn_index"] = raw.groupby("dialogue_id", sort=False).cumcount()

    result = raw[CANONICAL_COLUMNS].copy()
    result["dialogue_id"] = result["dialogue_id"].astype("string").str.strip()
    result["speaker"] = result["speaker"].astype("string").str.strip()
    result["text"] = result["text"].fillna("").astype(str).str.strip()
    result["turn_index"] = pd.to_numeric(result["turn_index"], errors="coerce")
    valid = (
        result["dialogue_id"].notna()
        & result["dialogue_id"].ne("")
        & result["speaker"].notna()
        & result["speaker"].ne("")
        & result["text"].ne("")
        & result["turn_index"].notna()
    )
    dropped = int((~valid).sum())
    result = result.loc[valid].copy()
    if result.empty:
        raise ValueError("No valid dialogue turns remain after removing blank or incomplete rows.")
    result["turn_index"] = result["turn_index"].astype(int)
    result = result.sort_values(
        ["dialogue_id", "turn_index"], kind="stable"
    ).reset_index(drop=True)

    if role is None:
        # Preserve canonical labels; otherwise use each dialogue's first two
        # participants in chronological order to define the role labels.
        role_maps: dict[str, dict[str, str]] = {}
        for dialogue_id, group in result.groupby("dialogue_id", sort=False):
            speakers = list(dict.fromkeys(group["speaker"].tolist()))
            if set(speakers).issubset({"initiator", "follower"}):
                role_maps[str(dialogue_id)] = {speaker: speaker for speaker in speakers}
            else:
                role_maps[str(dialogue_id)] = {
                    speaker: ("initiator" if i == 0 else "follower" if i == 1 else speaker)
                    for i, speaker in enumerate(speakers)
                }
        result["speaker"] = [
            role_maps[str(dialogue_id)][speaker]
            for dialogue_id, speaker in zip(result["dialogue_id"], result["speaker"])
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} turns to {output_path} ({dropped} incomplete rows dropped).")
    return result


def main() -> None:
    """Parse CLI arguments and normalize the requested input file."""
    parser = argparse.ArgumentParser(description="Normalize dialogue data to canonical turn CSV.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=["auto", "csv", "json"], default="auto")
    parser.add_argument("--id-column")
    parser.add_argument("--speaker-column")
    parser.add_argument("--turn-column")
    parser.add_argument("--text-column")
    parser.add_argument(
        "--role",
        choices=["initiator", "follower"],
        help="Set for a file containing turns for one speaker only, e.g. legacy role JSON.",
    )
    args = parser.parse_args()
    preprocess(
        args.input,
        args.output,
        args.format,
        args.id_column,
        args.speaker_column,
        args.turn_column,
        args.text_column,
        args.role,
    )


if __name__ == "__main__":
    main()
