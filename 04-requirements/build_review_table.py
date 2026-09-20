"""Join sentence evidence to STM document topics for human requirement review."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

_TOPIC_COLUMNS = ["post_id", "topic_id", "topic_name", "topic_prob_distribution"]
_REVIEW_COLUMNS = ["topic_id", "topic_name", "aspect", "sent_id", "post_id", "sentence", "context", "product_category", "n_words", "topic_prob_distribution", "requirement_candidate", "review_decision", "review_notes"]
_STAMP = re.compile(r"_\d{8}_\d{6}$")


def latest_stm_results(base: Path) -> Path | None:
    runs = [p for p in base.iterdir() if p.is_dir() and _STAMP.search(p.name) and (p / "stm_results.csv").exists()] if base.exists() else []
    return max(runs, key=lambda p: p.name) / "stm_results.csv" if runs else None


def join_stm_topics(sentences: pd.DataFrame, stm_results: pd.DataFrame | None) -> pd.DataFrame:
    if stm_results is None:
        out = sentences.copy()
        for column in _TOPIC_COLUMNS[1:]: out[column] = pd.NA
        return out
    missing = set(_TOPIC_COLUMNS) - set(stm_results.columns)
    if missing: raise ValueError(f"STM results missing columns: {sorted(missing)}")
    out = sentences.merge(stm_results[_TOPIC_COLUMNS], on="post_id", how="left", validate="many_to_one")
    if len(out) != len(sentences): raise AssertionError("STM join changed sentence row count")
    return out


def _topic_probability(value: object) -> float:
    try: return max(json.loads(value))
    except (TypeError, ValueError, json.JSONDecodeError): return float("-inf")


def make_review_table(linked: pd.DataFrame) -> pd.DataFrame:
    out = linked.copy()
    out["aspect"] = pd.NA
    out["requirement_candidate"] = pd.NA
    out["review_decision"] = pd.NA
    out["review_notes"] = pd.NA
    out["_rank"] = out.get("topic_prob_distribution", pd.Series(pd.NA, index=out.index)).map(_topic_probability)
    out = out.sort_values(["_rank", "n_words"], ascending=[False, False], kind="stable")
    for column in _REVIEW_COLUMNS:
        if column not in out: out[column] = pd.NA
    return out[_REVIEW_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stm-results")
    args = parser.parse_args()
    sentences = pd.read_csv(args.sentences)
    root = Path(__file__).resolve().parents[1]
    stm_path = Path(args.stm_results) if args.stm_results else latest_stm_results(root / "03-topic-modeling" / "data" / "output" / "youtube_doc" / "stm")
    stm = pd.read_csv(stm_path) if stm_path else None
    review = make_review_table(join_stm_topics(sentences, stm))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(output, index=False, encoding="utf-8")
    print(f"{len(review)} review rows -> {output}; stm_linked={stm is not None}")
    return 0


if __name__ == "__main__": raise SystemExit(main())