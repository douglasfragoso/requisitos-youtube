"""Global sentence NMF for the final YouTube topic pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "03-topic-modeling" / "configs" / "params.yaml"
SENTENCE_ROOT = ROOT / "02-sentences" / "data" / "output" / "youtube_sent"
OUTPUT_ROOT = ROOT / "03-topic-modeling" / "data" / "output" / "youtube_sent" / "nmf_global"
COLUMNS = ["sent_id", "post_id", "sentence", "context", "product_category",
           "n_words", "is_fragment", "is_dup_exact"]


def source_file() -> Path:
    files = sorted(SENTENCE_ROOT.glob("youtube_sent_*/corpus_sentencas.csv"))
    if not files:
        raise FileNotFoundError(f"corpus_sentencas.csv ausente em {SENTENCE_ROOT}")
    return files[-1]


def load_sentences(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, usecols=COLUMNS, dtype={"sent_id": str, "post_id": str})
    counts = {"source_rows": len(frame)}
    frame = frame.loc[~frame.is_fragment & ~frame.is_dup_exact].copy()
    frame = frame.dropna(subset=["sent_id", "post_id", "sentence", "context",
                                 "product_category"])
    frame = frame.loc[frame.n_words >= 5].reset_index(drop=True)
    counts["filtered_rows"] = len(frame)
    if not frame.sent_id.is_unique:
        raise ValueError("sent_id duplicado após filtragem")
    return frame, counts


def topic_examples(frame: pd.DataFrame, weights: np.ndarray, topic: int,
                   seed: int, count: int = 5) -> tuple[list[dict], list[dict]]:
    ids = np.flatnonzero((weights.argmax(axis=1) == topic) & (weights.sum(axis=1) > 0))
    ranked = ids[np.argsort(-weights[ids, topic], kind="stable")]

    def pick(indices: np.ndarray) -> list[dict]:
        selected = []
        seen_videos = set()
        for index in indices:
            row = frame.iloc[int(index)]
            if row.post_id in seen_videos:
                continue
            seen_videos.add(row.post_id)
            selected.append({"sent_id": row.sent_id, "post_id": row.post_id,
                             "category": row.product_category,
                             "sentence": row.sentence,
                             "context": row.context,
                             "weight": round(float(weights[index, topic]), 4)})
            if len(selected) == count:
                break
        return selected

    random_ids = np.random.default_rng(seed + topic).permutation(ids)
    return pick(ranked), pick(random_ids)


def run(path: Path, ks: list[int], seed: int, max_features: int,
        text_column: str) -> list[Path]:
    frame, counts = load_sentences(path)
    with PARAMS.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)["corpora"]["youtube_sent"]
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | set(cfg["stopwords_emojis"]))
    vectorizer = TfidfVectorizer(
        stop_words=stop_words, token_pattern=cfg["token_pattern"],
        min_df=20, max_df=0.5, max_features=max_features,
        sublinear_tf=True, dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(frame[text_column].fillna(""))
    active = np.asarray(matrix.getnnz(axis=1) > 0)
    counts["empty_after_vectorization"] = int((~active).sum())
    words = vectorizer.get_feature_names_out()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = []
    for k in ks:
        output = OUTPUT_ROOT / f"nmf_global_{text_column}_k{k}_{stamp}"
        output.mkdir(parents=True, exist_ok=False)
        model = NMF(n_components=k, init="nndsvda", random_state=seed,
                    max_iter=300, tol=1e-4)
        raw_weights = model.fit_transform(matrix[active])
        weights = np.zeros((len(frame), k), dtype=np.float32)
        sums = raw_weights.sum(axis=1)
        nonzero = sums > 0
        active_ids = np.flatnonzero(active)
        weights[active_ids[nonzero]] = raw_weights[nonzero] / sums[nonzero, None]
        dominant = weights.argmax(axis=1).astype(np.int16)
        dominant[weights.sum(axis=1) == 0] = -1
        strength = weights[np.arange(len(frame)), np.maximum(dominant, 0)]
        assignments = frame[["sent_id", "post_id", "sentence", "context",
                             "product_category"]].copy()
        assignments["topic_id"] = dominant
        assignments["topic_weight"] = strength
        assignments.to_csv(output / "nmf_results.csv", index=False)
        np.savez_compressed(output / "topic_weights.npz", sent_id=frame.sent_id.to_numpy(),
                            weights=weights)

        topics = []
        for topic in range(k):
            indices = np.argsort(-model.components_[topic])[:20]
            strong, random = topic_examples(frame, weights, topic, seed)
            topics.append({"topic_id": topic, "n_sentences": int((dominant == topic).sum()),
                           "top_terms": ", ".join(words[indices]),
                           "strong_examples": strong, "random_examples": random})
        (output / "topics.json").write_text(json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame([{key: row[key] for key in ("topic_id", "n_sentences", "top_terms")}
                      for row in topics]).to_csv(output / "topics.csv", index=False)
        with (output / "model.pkl").open("wb") as handle:
            pickle.dump({"model": model, "vectorizer": vectorizer}, handle)
        manifest = {"source": str(path), "source_size": path.stat().st_size,
                    "source_modified": path.stat().st_mtime, "seed": seed, "k": k,
                    "text_column": text_column, "target_column": "sentence",
                    "min_df": 20, "max_df": 0.5, "max_features": max_features,
                    "weight_semantics": "NMF activations normalized to sum to 1 per sentence; not calibrated probabilities",
                    "n_features": int(matrix.shape[1]), "n_nonzero": int(matrix.nnz),
                    "reconstruction_error": float(model.reconstruction_err_),
                    "n_iter": int(model.n_iter_), **counts}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"K={k}: {output} | {counts['filtered_rows']} frases | {matrix.shape[1]} termos | "
              f"erro={model.reconstruction_err_:.2f} | iterações={model.n_iter_}", flush=True)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    with PARAMS.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    final = params["final_sentence_pipeline"]
    parser.add_argument("--source", type=Path, default=source_file())
    parser.add_argument("--k", type=int, nargs="+", default=[final["global_k"]])
    parser.add_argument("--seed", type=int, default=params["seed"])
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--text-column", choices=["sentence", "context"],
                        default="sentence")
    args = parser.parse_args()
    run(args.source, args.k, args.seed, args.max_features, args.text_column)


if __name__ == "__main__":
    main()
