#!/usr/bin/env python3
"""Remove do objeto youtube_metadata campos que ja existem no topo do registro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DUPLICATED_FIELDS = {"title", "duration", "duration_string", "webpage_url"}


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy without metadata fields duplicated by the original schema."""
    cleaned = dict(record)
    metadata = record.get("youtube_metadata")
    if isinstance(metadata, dict):
        cleaned["youtube_metadata"] = {
            key: value for key, value in metadata.items() if key not in DUPLICATED_FIELDS
        }
    return cleaned


def clean_dataset(input_path: Path, output_path: Path) -> int:
    records = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("O dataset precisa ser um array JSON de objetos")
    cleaned = [clean_record(row) for row in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(cleaned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = clean_dataset(args.input, args.output)
    print(f"Dataset limpo: {count} registros -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
