"""Rebuild the historical STM-sentence to NMF-local review baseline.

The final sentence pipeline is run_topic_requirement_review.py. This command
keeps the earlier STM sentence experiment reproducible without changing it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from topic_requirement_review import discover_nmf_results, run_review


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--stm-results", type=Path, required=True)
    parser.add_argument("--nmf-root", type=Path, required=True)
    parser.add_argument("--topics", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=56669)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--n-random", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nmf_paths = discover_nmf_results(args.nmf_root, args.topics)
    sentences = pd.read_csv(args.sentences, usecols=[
        "sent_id", "post_id", "sentence", "context", "product_category", "source",
    ])
    stm = pd.read_csv(args.stm_results, usecols=[
        "post_id", "topic_id", "topic_name", "topic_prob_distribution",
    ])
    nmf = {topic: pd.read_csv(path, usecols=[
        "post_id", "topic_id", "topic_name", "topic_prob_distribution",
    ]) for topic, path in nmf_paths.items()}
    if sum(len(frame) for frame in nmf.values()) != args.expected_count:
        raise ValueError("contagem NMF difere da esperada")
    summary = run_review(sentences, stm, nmf, args.topics, args.output,
                         top_k=args.top_k, n_random=args.n_random, seed=args.seed)
    if summary["n_evidence"] != args.expected_count:
        raise ValueError("juncao alterou a contagem esperada")
    sources = [args.sentences, args.stm_results, *nmf_paths.values()]
    summary["sources"] = {str(path): sha256_file(path) for path in sources}
    (args.output / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "sources"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
