"""Builds ResultadoAnaliseDeSentimento.csv: one row per sentence of the corpus with its
metadata, NMF topics (when the sentence has one) and the sentiment assigned to it.

Whole-transcript columns (message_raw, message_punctuated) are left out: they would repeat
the full video text on every sentence row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SENTENCE_COLUMNS = ["sent_id", "post_id", "sent_idx", "sentence", "context", "n_words",
                    "is_fragment", "is_dup_exact", "punctuation_restored", "source",
                    "product_category", "channel", "upload_date", "year"]
SENTIMENT_COLUMNS = {"sentiment": "sentimento", "score": "confianca_sentimento",
                     "score_positiva": "score_positiva", "score_neutra": "score_neutra",
                     "score_negativa": "score_negativa"}
LOCAL_COLUMNS = ["sent_id", "local_topic_id", "local_topic_weight", "local_top_terms",
                 "lex_hit", "intent_hint"]


def build_result(sentences: pd.DataFrame, sentiment: pd.DataFrame,
                 global_topics: pd.DataFrame | None = None,
                 local_topics: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join everything on sent_id; the sentence corpus defines the rows and their order."""
    out = sentences[[c for c in SENTENCE_COLUMNS if c in sentences]].copy()
    if global_topics is not None:
        out = out.merge(global_topics, on="sent_id", how="left", validate="one_to_one")
        out.insert(out.columns.get_loc("global_topic_id") + 1, "tema_selecionado",
                   out["sent_id"].isin(local_topics["sent_id"]) if local_topics is not None else False)
    if local_topics is not None:
        out = out.merge(local_topics, on="sent_id", how="left", validate="one_to_one")
    scored = sentiment[["sent_id", *SENTIMENT_COLUMNS]].rename(columns=SENTIMENT_COLUMNS)
    out = out.merge(scored, on="sent_id", how="left", validate="one_to_one")
    if out["sentimento"].isna().any():
        raise ValueError("frases sem sentimento: o run de sentimento nao cobre este corpus")
    for column in ("global_topic_id", "local_topic_id"):
        if column in out:
            out[column] = out[column].astype("Int64")
    return out


def load_global_topics(run: Path) -> pd.DataFrame:
    results = pd.read_csv(run / "nmf_results.csv", usecols=["sent_id", "topic_id", "topic_weight"])
    terms = pd.read_csv(run / "topics.csv", usecols=["topic_id", "top_terms"])
    out = results.merge(terms, on="topic_id", how="left", validate="many_to_one")
    return out.rename(columns={"topic_id": "global_topic_id", "topic_weight": "global_topic_weight",
                               "top_terms": "global_top_terms"})


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?",
                        help="pasta sentimento_* (padrao: a mais recente com manifest.json)")
    parser.add_argument("--output", type=Path,
                        default=root / "05-sentiment" / "data" / "output" / "ResultadoAnaliseDeSentimento.csv")
    args = parser.parse_args()

    base = root / "05-sentiment" / "data" / "output"
    run = args.run or max((p for p in base.glob("sentimento_*") if (p / "manifest.json").exists()),
                          key=lambda p: p.name)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    sentences = pd.read_csv(manifest["input"], usecols=lambda c: c in SENTENCE_COLUMNS)
    sentiment = pd.read_csv(run / "sentencas_sentimento.csv")

    global_topics = local_topics = None
    if "evidence" in manifest:
        evidence = Path(manifest["evidence"])
        review = json.loads((evidence.parent / "metadata.json").read_text(encoding="utf-8"))
        global_topics = load_global_topics(Path(review["global_run"]))
        local_topics = pd.read_csv(evidence, usecols=LOCAL_COLUMNS)

    result = build_result(sentences, sentiment, global_topics, local_topics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"{len(result)} frases, {len(result.columns)} colunas -> {args.output}")
    print(result["sentimento"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
