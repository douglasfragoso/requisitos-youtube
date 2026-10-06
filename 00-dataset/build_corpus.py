"""Converte JSON de transcricoes e metadados do YouTube em CSV bruto.

    .venv/Scripts/python.exe 00-dataset/build_corpus.py \
        --input 00-dataset/transcricoes_youtube_metadados.json \
        --output 01-preprocessing/data/raw/youtube/youtube_reviews.csv

O filtro de idioma e feito posteriormente pelo 01-preprocessing, sobre a
transcricao; este modulo preserva o metadado de idioma como ``language_meta``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

COLUNAS = [
    "post_id", "message", "title", "channel", "upload_date", "year",
    "view_count", "like_count", "comment_count", "duration", "source",
    "language_meta", "product_category", "title_has_review",
]

# A ordem importa: o primeiro padrao que casar vence.
_PADROES: list[tuple[str, str]] = [
    ("tablet", r"\bipad\b|tablet|galaxy tab|surface pro\b|surface go\b|surface \d+\b|transformer(?: pad)?"),
    ("console", r"playstation|\bps[45]\b|xbox|nintendo switch|\bswitch\b|steam deck|rog ally|legion go|stadia"),
    ("vr", r"vision pro|meta quest|\bquest\b|\bvr\b|smart glasses|ray-?ban meta"),
    ("watch", r"\bwatch\b|\bband\b|fitbit|garmin|galaxy gear|galaxy ring|smartwatch|moto ?360"),
    ("headphone", r"headphone|earbud|airpods|\bbuds\b|headset|\bbeats\b|wh-?1000|\bxm[345]\b|beyerdynamic|audio technica|ath-m\d+|sony mdr"),
    ("monitor", r"\bmonitor\b|studio display|lg ultrafine|\bdisplay\b"),
    ("tv", r"\btv\b|\bqled\b|\boled tv\b|\bs95[cd]\b"),
    ("desktop", r"mac studio|mac mini|\bimac\b|mac pro\b|desktop|\bpc build|\bgaming pc\b|alienware aurora|\bnuc\b|corsair one"),
    ("camera", r"\bcamera\b|gopro|\bdji\b|drone|\brx\d+\b|sony nex|\bcanon\b|webcam"),
    ("phone", r"iphone|galaxy s\d|galaxy z|galaxy note|galaxy a\d|galaxy alpha|galaxy mega|\bpixel\b|oneplus|"
              r"\bphone\b|nothing phone|xiaomi|huawei|\bmoto\b|nexus|\bfold\b|\bflip\b|smartphone|"
              r"lumia|\boppo\b|\bhtc\b|\bnokia\b|\blg g\d\b|\blg optimus\b|\bmi \d|galaxy (?!book)|red hydrogen|motorola|droid|blackberry|surface duo|nextbit|pocophone|\bpoco\b|redmi|xperia|zenfone|one plus|realme|honor magic|\brazer\b|jovi|vivo"),
    ("laptop", r"laptop|notebook|macbook|\byoga\b|thinkpad|zenbook|vivobook|\bxps\b|surface laptop|surface book|"
               r"chromebook|\brog\b|legion|razer blade|\bomen\b|\baero\b|\bg1[456]\b|spectre|\benvy\b|"
               r"inspiron|latitude|\bswift\b|aspire|predator|pavilion|\bdell\b|\bhp\b|\bacer\b|\bmsi\b|"
               r"gigabyte|framework|matebook|ideapad|\bslim\b|alienware|galaxy book|\bgram\b|\btuf\b|"
               r"strix|zephyrus|\by\d{3}\b|\bx1 carbon\b|\bgl\d{3}\b|\bux\d{3}\b|\bg\d{3}\b|ultrabook|"
               r"triton|\bblade\b|\bstealth\b|\bx\d{2}\b|snapdragon x|\bmac\b|\blenovo\b|\basus\b|proart|aorus|eluktronics|helios|hasee|venom|pixelbook"),
]
_PADROES_COMPILADOS = [(categoria, re.compile(regex)) for categoria, regex in _PADROES]
_REVIEW = re.compile(r"\breview\b|\brant\b|\bhands[- ]on\b|\bunboxing\b|\bimpressions\b", re.I)


def detect_product_category(title: str) -> str:
    """Classifica pelo primeiro padrao de produto encontrado no titulo."""
    for categoria, regex in _PADROES_COMPILADOS:
        if regex.search(title.lower()):
            return categoria
    return "other"


def load_overrides(path: Path | str) -> dict[str, str]:
    """Le o mapeamento manual de IDs para categorias; arquivo ausente e valido."""
    arquivo = Path(path)
    if not arquivo.exists():
        return {}
    with arquivo.open(encoding="utf-8", newline="") as stream:
        return {
            linha["post_id"]: linha["product_category"]
            for linha in csv.DictReader(stream)
            if linha.get("post_id")
        }


def _iso(upload_date: str | None) -> str | None:
    if not upload_date or len(str(upload_date)) != 8:
        return None
    valor = str(upload_date)
    return f"{valor[:4]}-{valor[4:6]}-{valor[6:]}"


def build_corpus(records: list[dict], overrides: dict[str, str]) -> pd.DataFrame:
    """Produz a tabela bruta; registros sem transcricao ou metadados sao descartados."""
    rows = []
    for record in records:
        metadata = record.get("youtube_metadata") or {}
        message = (record.get("transcricao") or "").strip()
        post_id = metadata.get("id")
        if not message or not post_id:
            continue

        title = record.get("titulo") or ""
        upload_date = _iso(metadata.get("upload_date"))
        rows.append({
            "post_id": post_id,
            "message": message,
            "title": title,
            "channel": metadata.get("channel"),
            "upload_date": upload_date,
            "year": int(upload_date[:4]) if upload_date else None,
            "view_count": metadata.get("view_count"),
            "like_count": metadata.get("like_count"),
            "comment_count": metadata.get("comment_count"),
            "duration": record.get("duracao"),
            "source": record.get("fonte"),
            "language_meta": metadata.get("language"),
            "product_category": overrides.get(post_id) or detect_product_category(title),
            "title_has_review": bool(_REVIEW.search(title)),
        })

    corpus = pd.DataFrame(rows, columns=COLUNAS)
    duplicates = corpus["post_id"][corpus["post_id"].duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"post_id duplicado: {duplicates[:5]}")
    return corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overrides",
        default=str(Path(__file__).with_name("product_category_overrides.csv")),
    )
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as stream:
        records = json.load(stream)
    corpus = build_corpus(records, load_overrides(args.overrides))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output, index=False, encoding="utf-8")
    print(f"{len(records)} registros -> {len(corpus)} docs -> {output}")
    print(corpus["product_category"].value_counts().to_string())
    print(corpus["source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())