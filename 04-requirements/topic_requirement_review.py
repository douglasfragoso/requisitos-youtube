"""Lexical cues, reading-order rankings, blinded sample and scoring of human labels."""

import re

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
    frame = evidence[["sent_id", "lex_hit", "lex_n", "topic_score"]].copy()
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
    group_cols = ["global_topic_id", "local_topic_id"]
    for name in ("global_topic_name", "local_topic_name"):
        if name in merged:
            group_cols.append(name)
    merged["is_requirement"] = merged.requirement_candidate.eq("sim")
    by_topic = merged.groupby(group_cols, dropna=False).agg(
        n_reviewed=("review_id", "size"), n_sim=("is_requirement", "sum"),
    ).reset_index()
    return results, by_topic
