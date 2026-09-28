"""Run the predeclared BRETT T20 gate on three-sentence windows.

This repeats anchor selection with NMFregress on three independent 1,000-row
samples, then evaluates each anchor set on all 114,888 centered windows.
No category regression is reported by this gate.
"""

from pathlib import Path
import hashlib
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
    _run_brett_r, anchor_coverage_bound, build_tdm, context_windows,
    fracao_documentos_vazios,
)


def main() -> int:
    params = yaml.safe_load((TOPIC / "configs/params.yaml").read_text(encoding="utf-8"))
    config = params["brett"]["context_v2"]
    base = TOPIC / "data/output/youtube_sent/stm" / params["guided"]["gnmf"]["stm_run"]
    output = TOPIC / "data/output/youtube_sent/brett/c0_context_v2_20260927"
    output.mkdir(parents=True, exist_ok=True)
    original_frame = pd.read_csv(base / "stm_input.csv", usecols=["text", "post_id", "product_category"])
    stm_meta = json.loads((base / "stm_final.json").read_text(encoding="utf-8"))
    # Keep the original adjacency while constructing context. A sentence
    # removed by STM can still be the actual neighbor of a retained center.
    original_windows = context_windows(
        original_frame.text.fillna("").str.split().tolist(),
        original_frame.post_id.astype(str).tolist(), radius=int(config["radius"]),
    )
    kept_indices = original_frame.index.difference(stm_meta.get("docs_removed", []))
    frame = original_frame.loc[kept_indices].reset_index(drop=True)
    assert len(frame) == 114888
    documents = [original_windows[index] for index in kept_indices]
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=params["corpora"]["youtube_sent"]["stm_no_below"],
                               no_above=params["corpora"]["youtube_sent"]["stm_no_above"])
    allowed = set(dictionary.token2id)
    filtered = [[word for word in doc if word in allowed] for doc in documents]
    topics = int(config["topics"])
    package_source = TOPIC / "data/output/NMFregress-src"
    package_commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={package_source.as_posix()}", "rev-parse", "HEAD"],
        cwd=package_source, text=True,
    ).strip()
    summary = {
        "protocol": config["protocol"], "stm_run": base.name,
        "package_commit": package_commit,
        "stm_input_sha256": hashlib.sha256((base / "stm_input.csv").read_bytes()).hexdigest(),
        "radius": int(config["radius"]), "topics": topics,
        "context_source": "full_stm_input_before_docs_removed",
        "n_documents": len(filtered), "vocab_full": len(dictionary),
        "coverage_lower_bound": anchor_coverage_bound(dictionary, len(filtered), topics),
        "max_frac_vazias": float(params["brett"]["max_frac_vazias"]),
        "pilot_size": int(config["pilot_size"]), "pilots": [],
    }
    for seed in config["pilot_seeds"]:
        rng = np.random.default_rng(int(seed))
        indices = sorted(rng.choice(len(filtered), size=int(config["pilot_size"]), replace=False))
        sample = [filtered[index] for index in indices]
        sample_dictionary = Dictionary(sample)
        sample_dictionary.filter_extremes(no_below=3, no_above=0.5)
        if len(sample_dictionary) <= topics:
            raise RuntimeError("Vocabulário do piloto menor que T")
        tdm, vocab = build_tdm(sample, sample_dictionary)
        subdir = output / f"seed_{seed}"
        subdir.mkdir(exist_ok=True)
        np.savetxt(subdir / "tdm.csv", tdm, delimiter=",", fmt="%.0f")
        pd.Series(vocab).to_csv(subdir / "vocab.csv", index=False, header=False)
        result = _run_brett_r(subdir / "tdm.csv", subdir / "vocab.csv", subdir,
                              {"mode": "fit"}, seed=int(seed), topics=topics)
        theta = pd.read_csv(subdir / "theta.csv").to_numpy()
        sample_empty = fracao_documentos_vazios(theta)
        assert np.isclose(sample_empty, result["empty_fraction"])
        anchors = set(result["anchors"])
        full_empty = np.array([not anchors.intersection(doc) for doc in filtered])
        phi = pd.read_csv(subdir / "phi.csv")
        preview = []
        for anchor in result["anchors"]:
            for rank, row in enumerate(phi.nlargest(8, anchor).itertuples(index=False), start=1):
                preview.append({"anchor": anchor, "rank": rank, "word": row.word})
        pd.DataFrame(preview).to_csv(subdir / "topic_preview.csv", index=False)
        by_category = frame.assign(empty=full_empty).groupby("product_category").agg(
            n=("empty", "size"), frac_empty=("empty", "mean")
        ).reset_index()
        by_category.to_csv(subdir / "coverage_by_category.csv", index=False)
        row = {
            "seed": int(seed), "sample_vocab": len(sample_dictionary),
            "sample_empty": sample_empty, "full_empty": float(full_empty.mean()),
            "anchors": result["anchors"], "dir": str(subdir),
            "coverage_gate_pass": bool(full_empty.mean() <= summary["max_frac_vazias"]),
        }
        summary["pilots"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary["coverage_gate_pass"] = all(p["coverage_gate_pass"] for p in summary["pilots"])
    (output / "c0_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Gate C0:", "PASS" if summary["coverage_gate_pass"] else "FAIL", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
