"""Join aspect topics to sentence evidence for human requirements review."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LEXICON = {
    "pedido": (
        r"\bi wish\b", r"\bshould(?: have|'ve)\b", r"\bhoping for\b",
        r"\bwould have liked\b", r"\bmissing\b", r"\bi would like\b",
    ),
    "queixa": (
        r"\bdisappoint\w*\b", r"\bi (?:do not|don't) like\b",
        r"\bissue\b", r"\bproblem\b", r"\bcomplain\w*\b",
        r"\bannoy\w*\b", r"\black\b", r"\bdownside\b",
        r"\bdrawback\b", r"\bunfortunately\b",
    ),
}


def _dominant_weight(distribution, topic_id):
    weights = json.loads(distribution)
    if not isinstance(weights, list) or not 0 <= int(topic_id) < len(weights):
        raise ValueError("distribuicao de topicos invalida")
    return float(weights[int(topic_id)])


def join_topic_evidence(sentences, stm_results, nmf_by_stm_topic, expected_topics):
    """Join all canonical NMF aspect sentences to their original evidence."""
    missing = set(expected_topics) - set(nmf_by_stm_topic)
    if missing:
        raise ValueError(f"runs NMF faltando: {sorted(missing)}")
    nmf_parts = []
    for stm_topic in expected_topics:
        columns = ["post_id", "topic_id", "topic_prob_distribution"]
        if "topic_name" in nmf_by_stm_topic[stm_topic]:
            columns.append("topic_name")
        part = nmf_by_stm_topic[stm_topic][columns].copy()
        part["stm_topic_id_expected"] = stm_topic
        nmf_parts.append(part)
    nmf = pd.concat(nmf_parts, ignore_index=True).rename(columns={
        "post_id": "sent_id", "topic_id": "nmf_topic_id",
        "topic_prob_distribution": "nmf_distribution",
        "topic_name": "nmf_topic_name",
    })
    if not nmf.sent_id.is_unique:
        raise ValueError("sent_id duplicado nos runs NMF")
    if not sentences.sent_id.is_unique or not stm_results.post_id.is_unique:
        raise ValueError("sent_id duplicado nas entradas")
    selected = sentences.loc[sentences.sent_id.isin(nmf.sent_id)].copy()
    if len(selected) != len(nmf):
        raise ValueError("sent_id NMF ausente no corpus de sentencas")
    out = selected.merge(nmf, on="sent_id", how="left", validate="one_to_one", sort=False)
    stm_columns = ["post_id", "topic_id"]
    if "topic_name" in stm_results:
        stm_columns.append("topic_name")
    if "topic_prob_distribution" in stm_results:
        stm_columns.append("topic_prob_distribution")
    stm = stm_results[stm_columns].rename(columns={
        "post_id": "sent_id", "topic_id": "stm_topic_id",
        "topic_name": "stm_topic_name",
        "topic_prob_distribution": "stm_distribution",
    })
    out = out.merge(stm, on="sent_id", how="left", validate="one_to_one", sort=False)
    if out.stm_topic_id.isna().any() or not out.stm_topic_id.eq(out.stm_topic_id_expected).all():
        raise ValueError("sent_id ou tema STM divergente dos runs NMF")
    out["nmf_topic_weight"] = [
        _dominant_weight(dist, topic)
        for dist, topic in zip(out.nmf_distribution, out.nmf_topic_id)
    ]
    if "stm_distribution" in out:
        out["stm_topic_weight"] = [
            _dominant_weight(dist, topic)
            for dist, topic in zip(out.stm_distribution, out.stm_topic_id)
        ]
    else:
        out["stm_topic_weight"] = 1.0
    return out.drop(columns=["stm_topic_id_expected", "nmf_distribution",
                             "stm_distribution"], errors="ignore")


def score_lexical(sentences: pd.Series) -> pd.DataFrame:
    """Count frozen complaint/request cues without making human decisions."""
    rows = []
    for sentence in sentences.fillna("").astype(str):
        hits = {kind: [pattern for pattern in patterns
                       if re.search(pattern, sentence, flags=re.IGNORECASE)]
                for kind, patterns in LEXICON.items()}
        hint = "pedido" if hits["pedido"] else "queixa" if hits["queixa"] else "outro"
        rows.append({"lex_hit": hint != "outro",
                     "lex_n": sum(len(found) for found in hits.values()),
                     "intent_hint": hint})
    return pd.DataFrame(rows, index=sentences.index)


def build_rankings(evidence: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Rank the same sentence pool by topic, lexical cues and their combination."""
    if not evidence.sent_id.is_unique:
        raise ValueError("sent_id duplicado")
    frame = evidence[["sent_id", "nmf_topic_weight", "lex_hit", "lex_n"]].copy()
    frame["topic_score"] = frame.nmf_topic_weight * (
        evidence.stm_topic_weight if "stm_topic_weight" in evidence else 1.0
    )
    frame["_random_tie"] = np.random.default_rng(seed).permutation(len(frame))
    rules = {
        "topico": (["topic_score", "_random_tie", "sent_id"],
                   [False, True, True]),
        "lexical": (["lex_hit", "lex_n", "_random_tie", "sent_id"],
                    [False, False, True, True]),
        "topico_lexical": (["lex_hit", "lex_n", "topic_score",
                             "_random_tie", "sent_id"],
                           [False, False, False, True, True]),
        "aleatoria": (["_random_tie", "sent_id"], [True, True]),
    }
    ranked = []
    for name, (columns, ascending) in rules.items():
        ordered = frame.sort_values(columns, ascending=ascending, kind="stable")
        ranked.append(pd.DataFrame({"ranking": name,
                                    "rank": np.arange(1, len(frame) + 1),
                                    "sent_id": ordered.sent_id.to_numpy()}))
    return pd.concat(ranked, ignore_index=True)


def make_blind_sample(evidence, rankings, top_k=50, n_random=100, seed=42):
    """Export review text and method provenance in separate tables."""
    if top_k < 1 or n_random < 1:
        raise ValueError("tamanho da amostra invalido")
    top_methods = ("topico", "lexical", "topico_lexical")
    selected = []
    source = {}
    for method in top_methods:
        ids = rankings.loc[rankings.ranking.eq(method) & rankings["rank"].le(top_k), "sent_id"]
        if len(ids) != top_k:
            raise ValueError(f"ranking incompleto: {method}")
        for sent_id in ids:
            if sent_id not in source:
                selected.append(sent_id)
                source[sent_id] = method
    random_pool = rankings.loc[rankings.ranking.eq("aleatoria"), "sent_id"]
    random_ids = [sent_id for sent_id in random_pool if sent_id not in source][:n_random]
    if len(random_ids) != n_random:
        raise ValueError("amostra aleatoria sem candidatos suficientes")
    selected.extend(random_ids)
    source.update({sent_id: "aleatoria" for sent_id in random_ids})
    if not evidence.sent_id.is_unique:
        raise ValueError("sent_id duplicado")
    shuffled = np.random.default_rng(seed).permutation(selected).tolist()
    ids = pd.DataFrame({"review_id": np.arange(1, len(shuffled) + 1),
                        "sent_id": shuffled})
    detail = ids.merge(evidence, on="sent_id", validate="one_to_one", sort=False)
    if len(detail) != len(ids):
        raise ValueError("amostra perdeu sentencas")
    blind = detail[["review_id", "sentence", "context", "product_category"]].copy()
    blind["requirement_candidate"] = pd.NA
    blind["aspect_ref"] = pd.NA
    blind["review_notes"] = pd.NA
    ranks = rankings.pivot(index="sent_id", columns="ranking", values="rank")
    ranks = ranks.rename(columns={name: f"rank_{name}" for name in ranks.columns})
    origin = detail.drop(columns=["sentence", "context"]).merge(
        ranks, left_on="sent_id", right_index=True, validate="one_to_one",
    )
    origin["selection_source"] = origin.sent_id.map(source)
    return blind, origin


def precision_at_k(annotations, origin, ranking, k):
    """Compute precision only after all selected items have human labels."""
    if k < 1 or ranking not in {"topico", "lexical", "topico_lexical", "aleatoria"}:
        raise ValueError("ranking ou k invalido")
    if ranking == "aleatoria":
        chosen = origin.loc[origin.selection_source.eq("aleatoria")].head(k)
    else:
        chosen = origin.loc[origin[f"rank_{ranking}"].le(k)]
    if len(chosen) != k:
        raise ValueError("amostra de anotacao incompleta")
    labels = chosen[["review_id"]].merge(
        annotations[["review_id", "requirement_candidate"]],
        on="review_id", how="left", validate="one_to_one",
    ).requirement_candidate
    if labels.isna().any() or not labels.isin(["sim", "nao", "incerto"]).all():
        raise ValueError("anotacao humana ausente ou invalida")
    positives = int(labels.eq("sim").sum())
    proportion = positives / k
    z = 1.959963984540054
    denominator = 1 + z * z / k
    center = (proportion + z * z / (2 * k)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / k + z * z / (4 * k * k)) / denominator
    return {"ranking": ranking, "n": k, "positives": positives,
            "precision": proportion, "ci_low": float(center - half),
            "ci_high": float(center + half)}


def discover_nmf_results(nmf_root: Path, aspect_topics):
    """Require exactly one canonical NMF result for each STM aspect topic."""
    paths = {}
    for topic in aspect_topics:
        name = f"topic{int(topic):02d}"
        matches = sorted((nmf_root / name).glob(f"{name}_*/nmf_results.csv"))
        if len(matches) != 1:
            raise ValueError(f"{name}: esperado um run NMF, encontrados {len(matches)}")
        paths[int(topic)] = matches[0]
    return paths


def run_review(sentences, stm_results, nmf_by_stm_topic, expected_topics,
               output_dir: Path, top_k=50, n_random=100, seed=42):
    """Write the evidence and a blinded, unlabelled human-review sample."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = join_topic_evidence(sentences, stm_results, nmf_by_stm_topic,
                                   expected_topics)
    lexical = score_lexical(evidence.sentence)
    evidence = pd.concat([evidence.reset_index(drop=True), lexical.reset_index(drop=True)], axis=1)
    evidence["topic_score"] = evidence.stm_topic_weight * evidence.nmf_topic_weight
    rankings = build_rankings(evidence, seed=seed)
    blind, origin = make_blind_sample(evidence, rankings, top_k=top_k,
                                      n_random=n_random, seed=seed)
    evidence.to_csv(output_dir / "evidencias.csv", index=False, encoding="utf-8")
    rankings.to_csv(output_dir / "rankings.csv", index=False, encoding="utf-8")
    blind.to_csv(output_dir / "amostra_cega.csv", index=False, encoding="utf-8")
    origin.to_csv(output_dir / "amostra_origem.csv", index=False, encoding="utf-8")
    group_columns = ["stm_topic_id", "nmf_topic_id"]
    for name in ("stm_topic_name", "nmf_topic_name"):
        if name in evidence:
            group_columns.append(name)
    grouped = evidence.groupby(group_columns, dropna=False).agg(
        n=("sent_id", "size"), n_lex_hit=("lex_hit", "sum"),
        frac_lex_hit=("lex_hit", "mean"),
    ).reset_index()
    grouped.to_csv(output_dir / "sinais_por_topico.csv", index=False, encoding="utf-8")
    summary = {"n_evidence": int(len(evidence)), "n_lex_hit": int(evidence.lex_hit.sum()),
               "n_blind": int(len(blind)), "top_k": int(top_k),
               "n_random": int(n_random), "seed": int(seed),
               "review_status": "aguardando_anotacao_humana",
               "topic_score": "stm_topic_weight * nmf_topic_weight", "lexicon": LEXICON}
    (output_dir / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def score_annotated_sample(annotations, origin, top_k=50, n_random=100):
    """Summarize completed human labels; topic counts describe the pooled sample."""
    if not annotations.review_id.is_unique or not origin.review_id.is_unique:
        raise ValueError("review_id duplicado")
    merged = origin.merge(annotations[["review_id", "requirement_candidate"]],
                          on="review_id", how="left", validate="one_to_one")
    if merged.requirement_candidate.isna().any() or not merged.requirement_candidate.isin(
        ["sim", "nao", "incerto"]
    ).all():
        raise ValueError("anotacao humana incompleta")
    results = {name: precision_at_k(annotations, origin, name,
                                    n_random if name == "aleatoria" else top_k)
               for name in ("topico", "lexical", "topico_lexical", "aleatoria")}
    group_cols = ["stm_topic_id", "nmf_topic_id"] if "nmf_topic_id" in merged else ["stm_topic_id"]
    for name in ("stm_topic_name", "nmf_topic_name"):
        if name in merged:
            group_cols.append(name)
    merged["is_requirement"] = merged.requirement_candidate.eq("sim")
    by_topic = merged.groupby(group_cols, dropna=False).agg(
        n_reviewed=("review_id", "size"), n_sim=("is_requirement", "sum"),
    ).reset_index()
    return results, by_topic
