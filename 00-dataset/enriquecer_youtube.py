#!/usr/bin/env python3
"""Enriquece un dataset de transcricoes do YouTube com metadados publicos.

Uso:
    python enriquecer_youtube.py --input dataset.json --output dataset_com_metadados.json

Dependencia:
    python -m pip install -U yt-dlp
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


METADATA_FIELDS = (
    "id",
    "webpage_url",
    "title",
    "description",
    "channel",
    "channel_id",
    "uploader",
    "uploader_id",
    "upload_date",
    "timestamp",
    "duration",
    "duration_string",
    "view_count",
    "like_count",
    "comment_count",
    "categories",
    "tags",
    "language",
    "availability",
    "live_status",
    "thumbnail",
)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array, tolerating accidental text before the first '['."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    start = raw.find("[")
    if start < 0:
        raise ValueError(f"Nenhum array JSON encontrado em {path}")
    data = json.loads(raw[start:])
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError("O dataset precisa ser um array JSON de objetos")
    return data


def extract_video_id(url: str) -> str | None:
    """Extract a YouTube video id from watch, short, embed, live or youtu.be URLs."""
    if not isinstance(url, str) or not url.strip():
        return None
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":", 1)[0]
    path_parts = [part for part in parsed.path.split("/") if part]
    if host in {"youtu.be", "www.youtu.be"} and path_parts:
        return path_parts[0]
    if host.endswith("youtube.com"):
        query_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_id:
            return query_id
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            return path_parts[1]
    return None


def _metadata_from_info(info: dict[str, Any]) -> dict[str, Any]:
    return {field: info[field] for field in METADATA_FIELDS if field in info and info[field] is not None}


def merge_metadata(record: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a record with selected YouTube fields attached."""
    enriched = dict(record)
    enriched["youtube_metadata"] = _metadata_from_info(info)
    enriched["metadata_status"] = "ok"
    enriched.pop("metadata_error", None)
    return enriched


def _load_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "Dependencia ausente. Instale com: python -m pip install -U yt-dlp"
        ) from exc
    return yt_dlp


def fetch_metadata(url: str) -> dict[str, Any]:
    yt_dlp = _load_yt_dlp()
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("A resposta do YouTube nao trouxe metadados de video")
    return info


def _write_json_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def enrich_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    delay: float = 0.5,
    limit: int | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Fetch metadata for each record and persist progress after every video."""
    source = load_dataset(input_path)
    output = load_dataset(output_path) if resume and Path(output_path).exists() else [dict(item) for item in source]
    if len(output) != len(source):
        output = [dict(item) for item in source]

    processed = 0
    for index, record in enumerate(source):
        if limit is not None and processed >= limit:
            break
        current = output[index]
        if resume and current.get("metadata_status") == "ok":
            continue
        url = record.get("url")
        if not isinstance(url, str) or not url.strip():
            current["metadata_status"] = "skipped"
            current["metadata_error"] = "Registro sem URL"
        elif extract_video_id(url) is None:
            current["metadata_status"] = "skipped"
            current["metadata_error"] = "URL nao reconhecida como YouTube"
        else:
            try:
                output[index] = merge_metadata(record, fetch_metadata(url))
            except Exception as exc:  # one unavailable video must not stop the batch
                current["metadata_status"] = "error"
                current["metadata_error"] = str(exc)[:500]
        processed += 1
        _write_json_atomic(Path(output_path), output)
        if delay > 0 and processed < (limit or len(source)):
            time.sleep(delay)
        print(f"[{index + 1}/{len(source)}] {current.get('metadata_status', 'ok')}: {url}")
    _write_json_atomic(Path(output_path), output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Dataset JSON de entrada")
    parser.add_argument("--output", required=True, type=Path, help="JSON enriquecido de saida")
    parser.add_argument("--delay", type=float, default=0.5, help="Pausa entre consultas (segundos)")
    parser.add_argument("--limit", type=int, default=None, help="Processar no maximo N registros")
    parser.add_argument("--no-resume", action="store_true", help="Reprocessar registros ja concluidos")
    args = parser.parse_args()
    try:
        enrich_dataset(args.input, args.output, delay=max(0.0, args.delay), limit=args.limit, resume=not args.no_resume)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
