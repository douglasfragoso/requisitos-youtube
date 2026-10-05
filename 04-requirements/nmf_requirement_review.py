"""Prepare the final NMF-global to NMF-local sentence evidence for review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from topic_requirement_review import (LEXICON, build_rankings,
                                      make_blind_sample, score_lexical)


GLOBAL_COLUMNS = ["sent_id", "post_id", "sentence", "context",
                  "product_category", "topic_id", "topic_weight"]
LOCAL_COLUMNS = ["sent_id", "post_id", "sentence", "context",
                 "product_category", "global_topic_id", "global_topic_weight",
                 "local_topic_id", "local_topic_distribution"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_weight(value: str, topic_id: int, k: int) -> float:
    distribution = json.loads(value)
    if not isinstance(distribution, list) or len(distribution) != k:
        raise ValueError("distribuicao NMF local com tamanho incorreto")
    weights = np.asarray(distribution, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("distribuicao NMF local invalida")
    if topic_id == -1:
        if not np.isclose(weights.sum(), 0):
            raise ValueError("frase sem subtópico local tem peso positivo")
        return 0.0
    if not 0 <= topic_id < k or not np.isclose(weights.sum(), 1, atol=1e-5):
        raise ValueError("subtópico ou soma de pesos NMF local invalido")
    return float(weights[topic_id])


def load_nmf_evidence(global_run: Path, topics: list[int],
                      local_subdir: str = "restricted_gensim") -> tuple[pd.DataFrame, list[Path]]:
    """Join the selected global assignments to their canonical local NMF run."""
    global_run = Path(global_run)
    if not topics or len(topics) != len(set(topics)):
        raise ValueError("a lista de temas deve ser nao vazia e sem duplicatas")
    manifest_path = global_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("text_column") != "sentence":
        raise ValueError("o NMF global final deve usar a frase central")
    global_path = global_run / "nmf_results.csv"
    global_frame = pd.read_csv(global_path, usecols=GLOBAL_COLUMNS,
                               dtype={"sent_id": str, "post_id": str},
                               keep_default_na=False)
    if not global_frame.sent_id.is_unique:
        raise ValueError("sent_id duplicado no NMF global")
    selected = global_frame.loc[global_frame.topic_id.isin(topics)].copy()
    if not (selected.topic_weight.between(0, 1)).all():
        raise ValueError("peso do NMF global fora de [0,1]")

    global_terms = pd.read_csv(global_run / "topics.csv", usecols=["topic_id", "top_terms"])
    if not global_terms.topic_id.is_unique or not set(topics) <= set(global_terms.topic_id):
        raise ValueError("temas selecionados ausentes do export global")
    selected = selected.merge(global_terms.rename(columns={"top_terms": "global_top_terms"}),
                              on="topic_id", validate="many_to_one", sort=False)
    selected = selected.rename(columns={"topic_id": "global_topic_id",
                                        "topic_weight": "global_topic_weight"})

    parts, sources = [], [manifest_path, global_path, global_run / "topics.csv"]
    for topic in topics:
        local_dir = global_run / local_subdir / f"topic{topic:02d}"
        local_path = local_dir / "nmf_results.csv"
        local_manifest_path = local_dir / "manifest.json"
        local_topics_path = local_dir / "topics.csv"
        local_manifest = json.loads(local_manifest_path.read_text(encoding="utf-8"))
        k = int(local_manifest["selected_k"])
        if int(local_manifest["global_topic"]) != topic:
            raise ValueError(f"manifesto local diverge do tema global {topic}")
        part = pd.read_csv(local_path, usecols=LOCAL_COLUMNS,
                           dtype={"sent_id": str, "post_id": str},
                           keep_default_na=False)
        if not part.global_topic_id.eq(topic).all() or not part.sent_id.is_unique:
            raise ValueError(f"atribuicoes locais invalidas no tema {topic}")
        part["local_topic_weight"] = [
            _local_weight(value, int(local_topic), k)
            for value, local_topic in zip(part.local_topic_distribution, part.local_topic_id)
        ]
        local_terms = pd.read_csv(local_topics_path, usecols=["topic_id", "top_terms"])
        if len(local_terms) != k or not local_terms.topic_id.is_unique:
            raise ValueError(f"termos locais invalidos no tema {topic}")
        part = part.merge(local_terms.rename(columns={"topic_id": "local_topic_id",
                                               "top_terms": "local_top_terms"}),
                          on="local_topic_id", how="left", validate="many_to_one", sort=False)
        parts.append(part)
        sources.extend([local_path, local_manifest_path, local_topics_path])
    local = pd.concat(parts, ignore_index=True)
    if not local.sent_id.is_unique or len(local) != len(selected) or set(local.sent_id) != set(selected.sent_id):
        raise ValueError("cobertura local difere das frases selecionadas no NMF global")
    evidence = local.merge(selected[["sent_id", "post_id", "sentence", "context",
                                     "product_category", "global_topic_id", "global_topic_weight",
                                     "global_top_terms"]],
                           on="sent_id", validate="one_to_one", suffixes=("", "_global"), sort=False)
    for name in ("post_id", "sentence", "context", "product_category",
                 "global_topic_id"):
        if not evidence[name].eq(evidence[f"{name}_global"]).all():
            raise ValueError(f"dados locais divergem do NMF global: {name}")
    if not np.allclose(evidence.global_topic_weight,
                       evidence.global_topic_weight_global, atol=1e-6):
        raise ValueError("pesos globais divergem entre os runs")
    evidence = evidence.drop(columns=[f"{name}_global" for name in
                                      ("post_id", "sentence", "context", "product_category",
                                       "global_topic_id", "global_topic_weight")])
    evidence["topic_score"] = evidence.global_topic_weight * evidence.local_topic_weight
    return evidence, sources


def run_nmf_review(global_run: Path, topics: list[int], output_dir: Path,
                   *, local_subdir="restricted_gensim", expected_count: int | None = None,
                   top_k=50, n_random=100, seed=42) -> dict:
    evidence, sources = load_nmf_evidence(global_run, topics, local_subdir)
    if expected_count is not None and len(evidence) != expected_count:
        raise ValueError(f"esperadas {expected_count} frases; encontradas {len(evidence)}")
    evidence = pd.concat([evidence, score_lexical(evidence.sentence)], axis=1)
    rankings = build_rankings(evidence, seed=seed)
    blind, origin = make_blind_sample(evidence, rankings, top_k=top_k,
                                      n_random=n_random, seed=seed)
    grouped = evidence.groupby(["global_topic_id", "local_topic_id"], dropna=False).agg(
        n=("sent_id", "size"), n_lex_hit=("lex_hit", "sum"),
        frac_lex_hit=("lex_hit", "mean"),
    ).reset_index()
    by_global = evidence.groupby("global_topic_id").agg(
        n=("sent_id", "size"), n_lex_hit=("lex_hit", "sum"),
        frac_lex_hit=("lex_hit", "mean"),
    ).reset_index()
    summary = {
        "pipeline": "nmf_global_sentence_to_local_nmf_final",
        "global_run": str(Path(global_run).resolve()),
        "local_subdir": local_subdir,
        "selected_topics": list(topics),
        "n_evidence": int(len(evidence)),
        "n_unassigned_local": int(evidence.local_topic_id.eq(-1).sum()),
        "n_lex_hit": int(evidence.lex_hit.sum()),
        "n_blind": int(len(blind)),
        "top_k": int(top_k), "n_random": int(n_random), "seed": int(seed),
        "review_status": "aguardando_anotacao_humana",
        "topic_score": "global_topic_weight * local_topic_weight; zero for local_topic_id=-1",
        "weight_semantics": "normalized topic activations, not calibrated requirement probabilities",
        "lexicon": LEXICON,
        "sources_sha256": {str(path.resolve()): _sha256(path) for path in sources},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    evidence.to_csv(output_dir / "evidencias.csv", index=False, encoding="utf-8")
    rankings.to_csv(output_dir / "rankings.csv", index=False, encoding="utf-8")
    blind.to_csv(output_dir / "amostra_cega.csv", index=False, encoding="utf-8")
    origin.to_csv(output_dir / "amostra_origem.csv", index=False, encoding="utf-8")
    grouped.to_csv(output_dir / "sinais_por_subtopico.csv", index=False, encoding="utf-8")
    by_global.to_csv(output_dir / "sinais_por_topico_global.csv", index=False, encoding="utf-8")
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
