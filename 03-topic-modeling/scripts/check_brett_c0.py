"""Run a bounded BRETT pilot and the exact corpus-level C0 coverage bound."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from gensim.corpora import Dictionary

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "03-topic-modeling"
sys.path.insert(0, str(TOPIC / "notebooks"))
from _brett import (  # noqa: E402
    _run_brett_r, anchor_coverage_bound, build_tdm, fracao_documentos_vazios,
)

source = TOPIC / "data/output/youtube_sent/stm/youtube_sent_20260921_190124"
output = TOPIC / "data/output/youtube_sent/brett/c0_20260927"
output.mkdir(parents=True, exist_ok=True)
frame = pd.read_csv(source / "stm_input.csv")
meta = json.loads((source / "stm_final.json").read_text(encoding="utf-8"))
params = yaml.safe_load((TOPIC / "configs/params.yaml").read_text(encoding="utf-8"))
brett_cfg = params["brett"]
frame = frame.drop(index=meta.get("docs_removed", [])).reset_index(drop=True)
tokens = frame["text"].fillna("").str.split().tolist()
stm_topics = pd.read_csv(source / "stm_theta.csv").to_numpy().argmax(axis=1)
assert len(tokens) == len(stm_topics) == 114888
package_source = TOPIC / "data/output/NMFregress-src"
package_commit = None
if (package_source / ".git").exists():
    package_commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={package_source.as_posix()}",
         "rev-parse", "HEAD"], cwd=package_source, text=True
    ).strip()


def probe(name: str, docs: list[list[str]], no_below: int, topic_count: int) -> dict:
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below, no_above=0.5)
    bound = anchor_coverage_bound(dictionary, len(docs), topic_count)
    allowed = set(dictionary.token2id)
    rng = np.random.default_rng(42)
    sample_indices = sorted(rng.choice(len(docs), size=min(400, len(docs)), replace=False))
    sampled = [[word for word in docs[index] if word in allowed] for index in sample_indices]
    sample_dict = Dictionary(sampled)
    sample_dict.filter_extremes(no_below=3, no_above=0.5)
    assert len(sample_dict) > topic_count
    tdm, vocabulary = build_tdm(sampled, sample_dict)
    subdir = output / name
    subdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(subdir / "tdm.csv", tdm, delimiter=",", fmt="%.0f")
    pd.Series(vocabulary).to_csv(subdir / "vocab.csv", index=False, header=False)
    result = _run_brett_r(subdir / "tdm.csv", subdir / "vocab.csv", subdir,
                          {"mode": "fit"}, seed=42, topics=topic_count)
    theta = pd.read_csv(subdir / result["theta_csv"]).to_numpy()
    observed_sample = fracao_documentos_vazios(theta)
    assert np.isclose(observed_sample, result["empty_fraction"])
    anchors = set(result["anchors"])
    full_empty = sum(not anchors.intersection(doc) for doc in docs) / len(docs)
    summary = {"n_documents": len(docs), "n_sample": len(sampled),
               "vocab_full": len(dictionary), "vocab_pilot": len(sample_dict),
               "topics": topic_count, "minimum_empty_any_anchors": bound,
               "c0_provably_fails": bound > brett_cfg["max_frac_vazias"],
               "pilot_sample_empty": observed_sample,
               "pilot_anchors_on_full_corpus_empty": full_empty,
               "anchors": result["anchors"], "model_dir": str(subdir)}
    print(name, json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


all_results = {"stm_run": source.name, "nmfregress_commit": package_commit,
               "threshold": brett_cfg["max_frac_vazias"],
               "pilot_note": "400-document pilot; full-corpus lower bounds are exact"}
round_a_topics = int(meta["k"])
all_results["round_a"] = probe("rodada_a_t15_pilot", tokens, no_below=20,
                                topic_count=round_a_topics)
aspect = 6
subset = [doc for doc, topic in zip(tokens, stm_topics) if topic == aspect]
round_b_topics = max(brett_cfg["rodada_b"]["k_range"])
all_results["round_b_smallest"] = probe("rodada_b_topic06_t6_pilot", subset,
                                         no_below=5, topic_count=round_b_topics)
bounds = {}
for topic in [1, 3, 4, 6, 8, 10, 11, 12]:
    docs = [doc for doc, label in zip(tokens, stm_topics) if label == topic]
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bounds[str(topic)] = {"n_documents": len(docs),
                          "minimum_empty_t6": anchor_coverage_bound(dictionary, len(docs), round_b_topics)}
all_results["round_b_bounds"] = bounds
(output / "c0_summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
print("SUMMARY", output / "c0_summary.json", flush=True)
