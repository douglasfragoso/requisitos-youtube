"""Sentence-level sentiment (Positiva/Neutra/Negativa) for the YouTube sentence corpus.

Same model and label mapping as the earlier DataMining-ReviewFromYoutube project
(lxyuan/distilbert-base-multilingual-cased-sentiments-student). Sentiment is an
additional signal only: it never fills requirement_candidate nor enters the blind sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
LABEL_MAP = {"positive": "Positiva", "neutral": "Neutra", "negative": "Negativa"}
LABELS = ["Positiva", "Neutra", "Negativa"]
KEEP_COLUMNS = ["sent_id", "post_id", "sent_idx", "sentence", "n_words", "is_fragment",
                "is_dup_exact", "source", "product_category", "year", "channel"]
TOPIC_COLUMNS = ["sent_id", "global_topic_id", "local_topic_id", "global_topic_weight",
                 "local_topic_weight", "lex_hit", "intent_hint"]

# A classifier takes a batch of texts and returns, per text, every label with its score.
Classifier = Callable[[list[str]], list[list[dict]]]


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_record(scores: list[dict]) -> dict:
    """Best label plus the score of each label, with the labels mapped to Portuguese."""
    mapped = {LABEL_MAP[s["label"].lower()]: round(float(s["score"]), 4) for s in scores}
    best = max(mapped, key=mapped.get)
    return {"sentiment": best, "score": mapped[best],
            **{f"score_{label.lower()}": mapped.get(label, 0.0) for label in LABELS}}


class ScoreCache:
    """sha256(text) -> classifier output, so an interrupted run resumes where it stopped."""

    def __init__(self, path: Path | None):
        self.db = sqlite3.connect(path if path else ":memory:")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, scores TEXT)")

    def get_many(self, keys: list[str]) -> dict[str, list[dict]]:
        found = {}
        for start in range(0, len(keys), 900):
            chunk = keys[start:start + 900]
            rows = self.db.execute(
                f"SELECT key, scores FROM scores WHERE key IN ({','.join('?' * len(chunk))})", chunk)
            found.update({key: json.loads(scores) for key, scores in rows})
        return found

    def put_many(self, items: list[tuple[str, list[dict]]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?)",
                            [(key, json.dumps(scores)) for key, scores in items])
        self.db.commit()


def classify_sentences(df: pd.DataFrame, classifier: Classifier, cache: ScoreCache | None = None,
                       batch_size: int = 32, chunk_size: int = 2048, log=print) -> pd.DataFrame:
    """Classify each distinct sentence once and attach the result to every row."""
    if "sentence" not in df or "sent_id" not in df:
        raise ValueError("input needs sent_id and sentence columns")
    cache = cache or ScoreCache(None)
    texts = df["sentence"].fillna("").astype(str)
    unique = list(dict.fromkeys(texts))
    keys = {text: text_key(text) for text in unique}
    scored = cache.get_many(list(keys.values()))
    # Length-sorted batches pad to similar lengths; the scores do not depend on batch order.
    pending = sorted((text for text in unique if keys[text] not in scored), key=len)
    log(f"{len(unique)} frases unicas | {len(unique) - len(pending)} no cache | {len(pending)} a classificar")
    started = time.time()
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start:start + chunk_size]
        outputs = []
        for batch_start in range(0, len(chunk), batch_size):
            outputs.extend(classifier(chunk[batch_start:batch_start + batch_size]))
        items = [(keys[text], scores) for text, scores in zip(chunk, outputs)]
        cache.put_many(items)
        scored.update(items)
        done = start + len(chunk)
        log(f"[sentimento] {done}/{len(pending)} t={time.time() - started:.0f}s")
    records = pd.DataFrame([to_record(scored[keys[text]]) for text in texts], index=df.index)
    return pd.concat([df, records], axis=1)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    counts = pd.crosstab([df[c] for c in by], df["sentiment"]).reindex(columns=LABELS, fill_value=0)
    out = counts.copy()
    out.insert(0, "n", counts.sum(axis=1))
    for label in LABELS:
        out[f"frac_{label.lower()}"] = (counts[label] / out["n"]).round(4)
    out["saldo"] = (out["frac_positiva"] - out["frac_negativa"]).round(4)
    return out.reset_index()


def latest_run(base: Path, contains: str) -> Path:
    stamp = re.compile(r"_\d{8}_\d{6}(?:_revisao)?$")
    runs = [p for p in base.iterdir() if p.is_dir() and stamp.search(p.name) and (p / contains).exists()]
    if not runs:
        raise FileNotFoundError(f"nenhum run com {contains} em {base}")
    return max(runs, key=lambda p: p.stat().st_mtime) / contains


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hf_classifier(threads: int) -> tuple[Classifier, str]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import pipeline

    torch.set_num_threads(threads)
    path = snapshot_download(MODEL_NAME)
    pipe = pipeline("text-classification", model=path, tokenizer=path, truncation=True,
                    max_length=512, top_k=None)

    def classify(texts: list[str]) -> list[list[dict]]:
        return [list(scores) for scores in pipe(texts, batch_size=len(texts))]

    return classify, Path(path).name  # snapshot folder name = model commit hash


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="corpus_sentencas.csv (padrao: o run mais recente)")
    parser.add_argument("--evidence", type=Path,
                        help="evidencias.csv da etapa 04 (padrao: a revisao mais recente, se existir)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    source = args.input or latest_run(root / "02-sentences" / "data" / "output" / "youtube_sent",
                                      "corpus_sentencas.csv")
    evidence = args.evidence
    if evidence is None:
        try:
            evidence = latest_run(root / "04-requirements" / "data" / "output" / "topicos_requisitos",
                                  "evidencias.csv")
        except FileNotFoundError:
            evidence = None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output or root / "05-sentiment" / "data" / "output" / f"sentimento_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    cache_path = root / "05-sentiment" / "data" / "output" / "_cache" / "scores.sqlite"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"entrada: {source}\nevidencias: {evidence}\nsaida: {out_dir}", flush=True)
    sentences = pd.read_csv(source, usecols=lambda c: c in KEEP_COLUMNS)
    classifier, revision = hf_classifier(args.threads)
    result = classify_sentences(sentences, classifier, ScoreCache(cache_path), args.batch_size,
                                log=lambda msg: print(msg, flush=True))
    result.to_csv(out_dir / "sentencas_sentimento.csv", index=False, encoding="utf-8")

    filtered = result[~result["is_fragment"] & ~result["is_dup_exact"]]
    summarize(result, ["source"]).to_csv(out_dir / "resumo_por_fonte.csv", index=False)
    summarize(filtered, ["product_category"]).to_csv(out_dir / "resumo_por_categoria.csv", index=False)
    manifest = {
        "model": MODEL_NAME, "model_revision": revision, "label_map": LABEL_MAP,
        "text_column": "sentence", "input": str(source), "input_sha256": sha256_file(source),
        "n_sentences": int(len(result)), "n_unique_sentences": int(result["sentence"].nunique()),
        "n_filtered": int(len(filtered)),
        "filtered_definition": "is_fragment == False and is_dup_exact == False (resumos por categoria/tema)",
        "distribution_all": result["sentiment"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "distribution_filtered": filtered["sentiment"].value_counts().reindex(LABELS, fill_value=0).to_dict(),
        "note": "sinal auxiliar; nao confirma requisito nem entra na amostra cega",
    }
    if evidence is not None:
        topics = pd.read_csv(evidence, usecols=TOPIC_COLUMNS)
        joined = topics.merge(result[["sent_id", "sentiment", "score"]], on="sent_id", how="left",
                              validate="one_to_one")
        if joined["sentiment"].isna().any():
            raise ValueError("sent_id de evidencias.csv sem sentimento: corpus de frases diferente?")
        joined.to_csv(out_dir / "evidencias_sentimento.csv", index=False, encoding="utf-8")
        summarize(joined, ["global_topic_id"]).to_csv(out_dir / "resumo_por_tema.csv", index=False)
        summarize(joined, ["global_topic_id", "local_topic_id"]).to_csv(
            out_dir / "resumo_por_subtema.csv", index=False)
        summarize(joined, ["lex_hit"]).to_csv(out_dir / "resumo_por_lexico.csv", index=False)
        manifest.update({"evidence": str(evidence), "evidence_sha256": sha256_file(evidence),
                         "n_evidence": int(len(joined))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_sentences", "distribution_all",
                                               "distribution_filtered")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
