"""Reuse the historical Gensim BoW NMF protocol on selected global NMF topics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.models import CoherenceModel, Nmf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "03-topic-modeling" / "notebooks"))
from _helpers import (  # noqa: E402
    compute_topic_diversity,
    compute_doc_distributions,
    extract_topics_keywords,
    lemmatize_corpus,
    train_nmf,
)

PARAMS = ROOT / "03-topic-modeling" / "configs" / "params.yaml"


def pareto_choice(scores: pd.DataFrame) -> int:
    values = scores[["npmi", "diversity"]].to_numpy()
    dominated = np.array([
        np.any(np.all(values >= row, axis=1) & np.any(values > row, axis=1))
        for row in values
    ])
    front = scores.loc[~dominated]
    return int(front.sort_values(["npmi", "diversity", "k"],
                                 ascending=[False, False, True]).iloc[0].k)


def grid_scores(corpus_bow, dictionary, tokenized, ks, seed, top_n, output):
    rows = []
    for k in ks:
        model = Nmf(corpus=corpus_bow, id2word=dictionary, num_topics=k,
                    random_state=seed, passes=10)
        npmi = float(CoherenceModel(model=model, texts=tokenized,
                                    dictionary=dictionary, coherence="c_npmi",
                                    topn=top_n, processes=1).get_coherence())
        cv = float(CoherenceModel(model=model, texts=tokenized,
                                  dictionary=dictionary, coherence="c_v",
                                  topn=top_n, processes=1).get_coherence())
        words = {i: [word for word, _ in model.show_topic(i, topn=top_n)]
                 for i in range(k)}
        diversity = float(compute_topic_diversity(words, top_k=top_n))
        rows.append({"k": k, "npmi": npmi, "diversity": diversity, "cv": cv})
        pd.DataFrame(rows).to_csv(output / "grid_checkpoint.csv", index=False)
        print(f"  K={k}: npmi={npmi:.3f} diversidade={diversity:.3f}", flush=True)
    return pd.DataFrame(rows)


def run(global_run: Path, topics: list[int], seed: int) -> None:
    manifest = json.loads((global_run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["text_column"] != "sentence":
        raise ValueError("o run global deve usar a frase central")
    available = set(pd.read_csv(global_run / "topics.csv", usecols=["topic_id"]).topic_id)
    if not set(topics) <= available or len(set(topics)) != len(topics):
        raise ValueError("IDs de temas inválidos ou repetidos")
    frame = pd.read_csv(global_run / "nmf_results.csv",
                        dtype={"sent_id": str, "post_id": str})
    if not frame.sent_id.is_unique:
        raise ValueError("sent_id duplicado no run global")
    with PARAMS.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    cfg = params["corpora"]["youtube_sent"]
    local_cfg = params["nmf_restrito"]["youtube_sent"]
    no_below = int(local_cfg["no_below"])
    no_above = float(local_cfg["no_above"])
    ks = list(range(int(local_cfg["k_range"][0]), int(local_cfg["k_range"][1]) + 1))
    top_n = int(params["evaluation"]["top_n_keywords"])

    for global_topic in topics:
        subset = frame.loc[frame.topic_id == global_topic].reset_index(drop=True)
        output = global_run / "restricted_gensim" / f"topic{global_topic:02d}"
        if (output / "manifest.json").exists():
            print(f"tema global {global_topic}: resultado já concluído em {output}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        print(f"tema global {global_topic}: lematizando {len(subset)} frases", flush=True)
        tokenized, dictionary = lemmatize_corpus(
            subset.sentence.astype(str).tolist(), "en",
            {"nmf": {"no_below": no_below, "no_above": no_above}},
            model_key="nmf", extra_stopwords=cfg["stopwords_emojis"],
        )
        corpus_bow = [dictionary.doc2bow(tokens) for tokens in tokenized]
        diagnostics = grid_scores(corpus_bow, dictionary, tokenized, ks,
                                  seed, top_n, output)
        diagnostics.to_csv(output / "grid.csv", index=False)
        best_k = pareto_choice(diagnostics)
        model = train_nmf(corpus_bow, dictionary, k=best_k, seed=seed, passes=20)
        keywords = extract_topics_keywords(model, k=best_k, top_n=top_n)
        dominant, distributions = compute_doc_distributions(model, corpus_bow, best_k)
        dominant = [topic if bow and sum(dist) > 0 else -1
                    for topic, bow, dist in zip(dominant, corpus_bow, distributions)]
        assignments = subset[["sent_id", "post_id", "sentence", "context",
                              "product_category", "topic_weight"]].rename(
                                  columns={"topic_weight": "global_topic_weight"}).copy()
        assignments["global_topic_id"] = global_topic
        assignments["local_topic_id"] = dominant
        assignments["local_topic_distribution"] = [json.dumps(row) for row in distributions]
        assignments.to_csv(output / "nmf_results.csv", index=False)
        topic_rows = [{"topic_id": topic, "top_terms": ", ".join(words),
                       "n_sentences": int((assignments.local_topic_id == topic).sum())}
                      for topic, words in keywords.items()]
        pd.DataFrame(topic_rows).to_csv(output / "topics.csv", index=False)
        model.save(str(output / "model.gensim"))
        details = {"global_run": str(global_run), "global_topic": global_topic,
                   "seed": seed, "selected_k": best_k, "k_range": ks,
                   "n_sentences": len(subset), "vocabulary_size": len(dictionary),
                   "empty_bow": sum(not row for row in corpus_bow),
                   "unassigned": sum(topic < 0 for topic in dominant),
                   "method": "Gensim Nmf on spaCy lemmas, raw BoW counts; grid passes=10, final passes=20; Pareto NPMI x diversity"}
        (output / "manifest.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"tema global {global_topic}: K={best_k} | {len(dictionary)} termos | {output}", flush=True)


def repair_empty_assignments(global_run: Path, topics: list[int]) -> None:
    """Correct outputs made before empty BoW rows were marked unassigned."""
    for global_topic in topics:
        output = global_run / "restricted_gensim" / f"topic{global_topic:02d}"
        assignments = pd.read_csv(output / "nmf_results.csv")
        empty = assignments.local_topic_distribution.map(
            lambda value: sum(json.loads(value)) == 0
        )
        assignments.loc[empty, "local_topic_id"] = -1
        assignments.to_csv(output / "nmf_results.csv", index=False)
        topics_frame = pd.read_csv(output / "topics.csv")
        counts = assignments.local_topic_id.value_counts()
        topics_frame["n_sentences"] = topics_frame.topic_id.map(counts).fillna(0).astype(int)
        topics_frame.to_csv(output / "topics.csv", index=False)
        manifest_path = output / "manifest.json"
        details = json.loads(manifest_path.read_text(encoding="utf-8"))
        details["unassigned"] = int(empty.sum())
        manifest_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"tema global {global_topic}: {int(empty.sum())} frases sem palavras → -1", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    with PARAMS.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    final = params["final_sentence_pipeline"]
    default_run = (ROOT / "03-topic-modeling/data/output/youtube_sent/nmf_global"
                   / final["global_run"])
    parser.add_argument("global_run", type=Path, nargs="?", default=default_run)
    parser.add_argument("--topics", type=int, nargs="+", default=final["selected_topics"])
    parser.add_argument("--seed", type=int, default=params["seed"])
    parser.add_argument("--repair-existing", action="store_true")
    args = parser.parse_args()
    if args.repair_existing:
        repair_empty_assignments(args.global_run, args.topics)
    else:
        run(args.global_run, args.topics, args.seed)


if __name__ == "__main__":
    main()
