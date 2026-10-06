"""Build the final NMF-global to NMF-local blinded requirements review.

With no arguments, uses the frozen run and selected topics in
03-topic-modeling/configs/params.yaml::final_sentence_pipeline.
"""

import argparse
import json
from pathlib import Path

import yaml

from nmf_requirement_review import run_nmf_review


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "03-topic-modeling/configs/params.yaml"


def main():
    with PARAMS.open(encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    cfg = params["final_sentence_pipeline"]
    if not cfg.get("global_run") or not cfg.get("selected_topics"):
        raise ValueError("preencha global_run e selected_topics em "
                         "params.yaml::final_sentence_pipeline")
    default_run = (ROOT / "03-topic-modeling/data/output/youtube_sent/nmf_global"
                   / cfg["global_run"])
    default_output = (ROOT / "04-requirements/data/output/topicos_requisitos"
                      / (cfg.get("review_output") or f"{cfg['global_run']}_revisao"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-run", type=Path, default=default_run)
    parser.add_argument("--topics", type=int, nargs="+", default=cfg["selected_topics"])
    parser.add_argument("--local-subdir", default=cfg["local_subdir"])
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--expected-count", type=int, default=cfg["expected_sentences"])
    parser.add_argument("--top-k", type=int, default=cfg["top_k"])
    parser.add_argument("--n-random", type=int, default=cfg["n_random"])
    parser.add_argument("--seed", type=int, default=params["seed"])
    args = parser.parse_args()

    summary = run_nmf_review(
        args.global_run, args.topics, args.output,
        local_subdir=args.local_subdir, expected_count=args.expected_count,
        top_k=args.top_k, n_random=args.n_random, seed=args.seed,
    )
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "sources_sha256"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
