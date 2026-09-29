"""Run BRETT context v3 coverage pilots with a filtered global vocabulary."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

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
    brett_cfg = params["brett"]
    config = brett_cfg["context_v3"]
    stm_run = params["guided"]["gnmf"]["stm_run"]
    stm_dir = TOPIC / "data/output/youtube_sent/stm" / stm_run
    output = TOPIC / "data/output/youtube_sent/brett/c0_context_v3_filtered_vocabulary_20260928"
    output.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(stm_dir / "stm_input.csv", usecols=["text", "post_id", "product_category"])
    stm_meta = json.loads((stm_dir / "stm_final.json").read_text(encoding="utf-8"))
    windows = context_windows(original.text.fillna("").str.split().tolist(),
                              original.post_id.astype(str).tolist(),
                              radius=int(config["radius"]))
    if len(windows) != len(original):
        raise AssertionError("Esperada uma janela de contexto por frase")
    kept_indices = original.index.difference(stm_meta.get("docs_removed", []))
    frame = original.loc[kept_indices].reset_index(drop=True)
    documents = [windows[index] for index in kept_indices]
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=int(config["no_below"]),
                               no_above=float(config["no_above"]))
    allowed = set(dictionary.token2id)
    filtered = [[word for word in doc if word in allowed] for doc in documents]
    topics = int(config["topics"])
    if topics >= len(dictionary):
        raise RuntimeError(f"Vocabulário ({len(dictionary)}) menor que T={topics}")
    package_source = TOPIC / "data/output/NMFregress-src"
    package_commit = subprocess.check_output(
        ["git", "-c", f"safe.directory={package_source.as_posix()}", "rev-parse", "HEAD"],
        cwd=package_source, text=True,
    ).strip()
    summary = {
        "protocol": config["protocol"], "stm_run": stm_run,
        "package_commit": package_commit,
        "stm_input_sha256": hashlib.sha256((stm_dir / "stm_input.csv").read_bytes()).hexdigest(),
        "radius": int(config["radius"]), "topics": topics,
        "context_source": "full_stm_input_before_docs_removed",
        "n_documents": len(filtered), "vocab_full": len(dictionary),
        "no_below": int(config["no_below"]), "no_above": float(config["no_above"]),
        "coverage_lower_bound": anchor_coverage_bound(dictionary, len(filtered), topics),
        "max_frac_vazias": float(brett_cfg["max_frac_vazias"]),
        "pilot_size": int(config["pilot_size"]), "pilots": [],
    }
    if summary["coverage_lower_bound"] > summary["max_frac_vazias"]:
        (output / "c0_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"coverage_gate_pass": False, **summary}, ensure_ascii=False), flush=True)
        return 0

    for seed in config["pilot_seeds"]:
        seed = int(seed)
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(filtered), size=int(config["pilot_size"]), replace=False))
        sample = [filtered[index] for index in indices]
        sample_dictionary = Dictionary(sample)
        sample_dictionary.filter_extremes(
            no_below=int(config["sample_no_below"]),
            no_above=float(config["sample_no_above"]),
        )
        if len(sample_dictionary) <= topics:
            raise RuntimeError(f"Vocabulario do piloto ({len(sample_dictionary)}) <= T={topics}")
        tdm, vocab = build_tdm(sample, sample_dictionary)
        subdir = output / f"seed_{seed}"
        subdir.mkdir(exist_ok=True)
        np.savetxt(subdir / "tdm.csv", tdm, delimiter=",", fmt="%.0f")
        pd.Series(vocab).to_csv(subdir / "vocab.csv", index=False, header=False)
        result = _run_brett_r(subdir / "tdm.csv", subdir / "vocab.csv", subdir,
                              {"mode": "fit", "rscript_path": params["stm"]["rscript_path"]},
                              seed=seed, topics=topics)
        theta = pd.read_csv(subdir / "theta.csv").to_numpy()
        if theta.shape[1] != len(sample):
            raise ValueError("Theta BRETT nao alinha com as frases-piloto")
        sample_empty = fracao_documentos_vazios(theta)
        if not np.isclose(sample_empty, result["empty_fraction"]):
            raise AssertionError("Taxa vazia reportada diverge da theta")
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
            "seed": seed, "sample_vocab": len(sample_dictionary),
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
