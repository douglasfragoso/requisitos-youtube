"""Build a provenance-preserving sentence corpus from YouTube transcripts."""
from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import pandas as pd
import spacy


class PunctuationModel(Protocol):
    def restore_punctuation(self, text: str) -> str: ...


_REQUIRED = {"post_id", "message", "source", "product_category"}
_BASE_COLUMNS = [
    "sent_id", "post_id", "sent_idx", "sentence", "context", "message_raw",
    "message_punctuated", "punctuation_restored", "n_words", "is_fragment",
    "is_dup_exact", "upload_date", "year", "channel", "source", "product_category",
]


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])


def _sentences(text: str) -> list[str]:
    return [sent.text.strip() for sent in _nlp()(text).sents if sent.text.strip()]


def build_sentence_corpus(
    df: pd.DataFrame,
    punctuation_model: PunctuationModel | None = None,
    context_window: int = 1,
) -> pd.DataFrame:
    """Segment every input document without crossing document boundaries."""
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if context_window < 0:
        raise ValueError("context_window must be non-negative")
    if df["post_id"].duplicated().any():
        raise ValueError("post_id must be unique in the document corpus")

    captions = df["source"].eq("legenda")
    if captions.any() and punctuation_model is None:
        raise RuntimeError("punctuation_model is required when source='legenda' is present")

    records: list[dict] = []
    passthrough = [c for c in ("upload_date", "year", "channel", "source", "product_category") if c in df.columns]
    for row in df.to_dict("records"):
        raw = str(row["message"])
        restored = row.get("source") == "legenda"
        punctuated = punctuation_model.restore_punctuation(raw) if restored else raw
        sentences = _sentences(punctuated)
        for index, sentence in enumerate(sentences):
            lo, hi = max(0, index - context_window), min(len(sentences), index + context_window + 1)
            rec = {
                "sent_id": f"{row['post_id']}_{index}",
                "post_id": row["post_id"],
                "sent_idx": index,
                "sentence": sentence,
                "context": " ".join(sentences[lo:hi]),
                "message_raw": raw,
                "message_punctuated": punctuated,
                "punctuation_restored": restored,
                "n_words": len(sentence.split()),
                "is_fragment": len(sentence.split()) < 5,
            }
            rec.update({key: row.get(key) for key in passthrough})
            records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(columns=_BASE_COLUMNS)
    out["is_dup_exact"] = out.duplicated(subset=["sentence"], keep=False)
    if not out["sent_id"].is_unique:
        raise ValueError("sent_id must be unique")
    for column in _BASE_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out[_BASE_COLUMNS]

def resolve_latest_input(base: Path, contains: str = "corpus_limpo.csv") -> Path:
    """Return the newest stamped run directory containing the requested artifact."""
    stamp = __import__("re").compile(r"_\d{8}_\d{6}$")
    runs = [p for p in base.iterdir() if p.is_dir() and stamp.search(p.name) and (p / contains).exists()]
    if not runs:
        raise FileNotFoundError(f"no stamped run with {contains} below {base}")
    return max(runs, key=lambda p: p.name)


class _LocalPunctuationModel:
    """Compatibility adapter for deepmultilingualpunctuation on Transformers 5."""
    def __init__(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import pipeline
        path = snapshot_download("oliverguhr/fullstop-punctuation-multilang-large", local_files_only=True)
        self.pipe = pipeline("token-classification", model=path, aggregation_strategy="none")

    def restore_punctuation(self, text: str) -> str:
        words = re.sub(r"(?<!\d)[.,;:!?](?!\d)", "", text).split()
        if not words:
            return ""
        tagged = []
        for batch in (words[i:i + 230] for i in range(0, len(words), 225)):
            rendered = " ".join(batch)
            results = self.pipe(rendered)
            position = result_index = 0
            for word in batch:
                position += len(word) + 1
                label = "0"
                while result_index < len(results) and position > results[result_index]["end"]:
                    label = results[result_index]["entity"]
                    result_index += 1
                tagged.append((word, label))
        return "".join(word + (" " if label == "0" else label + " ") for word, label in tagged).strip()


def _punctuation_model() -> PunctuationModel:
    return _LocalPunctuationModel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--context-window", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = Path(args.input) if args.input else resolve_latest_input(root / "01-preprocessing" / "data" / "output" / "youtube") / "corpus_limpo.csv"
    destination = Path(args.output) if args.output else root / "02-sentences" / "data" / "output" / "youtube_sent" / f"youtube_sent_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}" / "corpus_sentencas.csv"
    documents = pd.read_csv(source).drop(columns=["message_raw"], errors="ignore")
    model = _punctuation_model() if documents["source"].eq("legenda").any() else None
    sentences = build_sentence_corpus(documents, punctuation_model=model, context_window=args.context_window)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sentences.to_csv(destination, index=False, encoding="utf-8")
    print(f"{len(documents)} docs -> {len(sentences)} sentences -> {destination}")
    print(sentences.groupby(["source", "punctuation_restored"]).size().to_string())
    print(f"fragments={int(sentences['is_fragment'].sum())} duplicates={int(sentences['is_dup_exact'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())