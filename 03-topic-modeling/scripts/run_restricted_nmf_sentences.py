"""Explore TF-IDF local NMF inside global sentence topics.

This alternative is not the frozen final pipeline. The final local model uses
run_restricted_nmf_gensim.py and its outputs feed nmf_requirement_review.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from run_global_nmf_sentences import PARAMS, topic_examples


def run(global_run: Path, topics: list[int], ks: list[int], seed: int,
        max_features: int) -> None:
    manifest = json.loads((global_run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["text_column"] != "sentence":
        raise ValueError("esta etapa exige um run global baseado na frase central")
    global_topics = pd.read_csv(global_run / "topics.csv", usecols=["topic_id"])
    valid = set(global_topics.topic_id)
    unknown = set(topics) - valid
    if unknown:
        raise ValueError(f"temas globais ausentes: {sorted(unknown)}")
    if len(set(topics)) != len(topics):
        raise ValueError("IDs de temas repetidos")
    frame = pd.read_csv(global_run / "nmf_results.csv", dtype={"sent_id": str, "post_id": str})
    if not frame.sent_id.is_unique:
        raise ValueError("sent_id duplicado no run global")
    with PARAMS.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)["corpora"]["youtube_sent"]
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | set(cfg["stopwords_emojis"]))

    for global_topic in topics:
        subset = frame.loc[frame.topic_id == global_topic].reset_index(drop=True)
        if len(subset) < 30:
            raise ValueError(f"tema {global_topic} tem somente {len(subset)} frases")
        vectorizer = TfidfVectorizer(
            stop_words=stop_words, token_pattern=cfg["token_pattern"],
            min_df=5, max_df=0.5, max_features=max_features,
            sublinear_tf=True, dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(subset.sentence.fillna(""))
        active = np.asarray(matrix.getnnz(axis=1) > 0)
        active_ids = np.flatnonzero(active)
        words = vectorizer.get_feature_names_out()
        for k in ks:
            output = global_run / "restricted" / f"topic{global_topic:02d}" / f"k{k}"
            if output.exists():
                raise FileExistsError(f"saída existente: {output}")
            output.mkdir(parents=True)
            model = NMF(n_components=k, init="nndsvda", random_state=seed,
                        max_iter=300, tol=1e-4)
            raw = model.fit_transform(matrix[active])
            weights = np.zeros((len(subset), k), dtype=np.float32)
            sums = raw.sum(axis=1)
            nonzero = sums > 0
            weights[active_ids[nonzero]] = raw[nonzero] / sums[nonzero, None]
            dominant = weights.argmax(axis=1).astype(np.int16)
            dominant[weights.sum(axis=1) == 0] = -1
            strength = weights[np.arange(len(subset)), np.maximum(dominant, 0)]
            assignments = subset[["sent_id", "post_id", "sentence", "context",
                                  "product_category", "topic_weight"]].rename(
                                      columns={"topic_weight": "global_topic_weight"})
            assignments = assignments.copy()
            assignments["global_topic_id"] = global_topic
            assignments["local_topic_id"] = dominant
            assignments["local_topic_weight"] = strength
            assignments.to_csv(output / "nmf_results.csv", index=False)
            np.savez_compressed(output / "topic_weights.npz",
                                sent_id=subset.sent_id.to_numpy(), weights=weights)
            local_topics = []
            for topic in range(k):
                indices = np.argsort(-model.components_[topic])[:20]
                strong, random = topic_examples(subset, weights, topic, seed)
                local_topics.append({"topic_id": topic,
                                     "n_sentences": int((dominant == topic).sum()),
                                     "top_terms": ", ".join(words[indices]),
                                     "strong_examples": strong,
                                     "random_examples": random})
            (output / "topics.json").write_text(
                json.dumps(local_topics, ensure_ascii=False, indent=2), encoding="utf-8")
            pd.DataFrame([{key: row[key] for key in ("topic_id", "n_sentences", "top_terms")}
                          for row in local_topics]).to_csv(output / "topics.csv", index=False)
            with (output / "model.pkl").open("wb") as handle:
                pickle.dump({"model": model, "vectorizer": vectorizer}, handle)
            details = {"global_run": str(global_run), "global_topic": global_topic,
                       "k": k, "seed": seed, "n_sentences": len(subset),
                       "n_unassigned": int((dominant < 0).sum()),
                       "vocabulary_size": int(matrix.shape[1]),
                       "reconstruction_error": float(model.reconstruction_err_),
                       "n_iter": int(model.n_iter_),
                       "weight_semantics": "NMF activations normalized per sentence; not probabilities"}
            (output / "manifest.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
            print(f"tema global {global_topic}, K local {k}: {output} | "
                  f"{len(subset)} frases | {matrix.shape[1]} termos", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("global_run", type=Path)
    parser.add_argument("--topics", type=int, nargs="+", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=12000)
    args = parser.parse_args()
    run(args.global_run, args.topics, args.k, args.seed, args.max_features)


if __name__ == "__main__":
    main()
