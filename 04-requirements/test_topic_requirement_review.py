import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import topic_requirement_review as review  # noqa: E402


def test_lexical_signal_finds_requests_and_complaints_without_marking_praise():
    scored = review.score_lexical(pd.Series([
        "I wish it had another USB port.",
        "The fan is annoying and noisy.",
        "I absolutely love it.",
        "This issue is disappointing.",
    ]))
    assert scored.lex_hit.tolist() == [True, True, False, True]
    assert scored.intent_hint.tolist() == ["pedido", "queixa", "outro", "queixa"]
    assert scored.lex_n.iloc[3] == 2


def test_rankings_keep_same_candidates_and_put_topic_and_lexical_first():
    evidence = pd.DataFrame([
        {"sent_id": "a", "topic_score": .9, "lex_hit": False, "lex_n": 0},
        {"sent_id": "b", "topic_score": .6, "lex_hit": True, "lex_n": 1},
        {"sent_id": "c", "topic_score": .7, "lex_hit": True, "lex_n": 1},
    ])
    first = review.build_rankings(evidence, seed=42)
    second = review.build_rankings(evidence, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert set(first.ranking) == {"topico", "lexical", "topico_lexical", "aleatoria"}
    assert first.loc[first.ranking.eq("topico")].sent_id.iloc[0] == "a"
    assert first.loc[first.ranking.eq("topico_lexical")].sent_id.iloc[0] == "c"
    assert all(part.sent_id.is_unique and set(part.sent_id) == {"a", "b", "c"}
               for _, part in first.groupby("ranking"))


def test_topic_weight_ties_are_seeded_random_not_sentence_id_order():
    evidence = pd.DataFrame({"sent_id": list("abcdefgh"),
                             "topic_score": [1.0] * 8,
                             "lex_hit": [False] * 8, "lex_n": [0] * 8})
    order = review.build_rankings(evidence, seed=42)
    ranked = order.loc[order.ranking.eq("topico"), "sent_id"].tolist()
    assert ranked != list("abcdefgh")
    assert ranked == review.build_rankings(evidence, seed=42).loc[
        lambda frame: frame.ranking.eq("topico"), "sent_id"
    ].tolist()


def test_blind_sample_keeps_origin_separate_and_random_outside_top_lists():
    evidence = pd.DataFrame([
        {"sent_id": f"v_{i}", "sentence": f"Sentence {i}", "context": f"Context {i}",
         "product_category": "laptop", "global_topic_id": 10, "local_topic_id": 0,
         "topic_score": 1 - i / 10, "lex_hit": i in {2, 3},
         "lex_n": int(i in {2, 3}), "intent_hint": "queixa" if i in {2, 3} else "outro"}
        for i in range(8)
    ])
    rankings = review.build_rankings(evidence, seed=42)
    blind, origin = review.make_blind_sample(evidence, rankings, top_k=2, n_random=2, seed=42)
    assert blind.review_id.is_unique and origin.review_id.is_unique
    assert set(blind.review_id) == set(origin.review_id)
    assert blind.requirement_candidate.isna().all()
    assert set(blind.columns) == {"review_id", "sentence", "context", "product_category",
                                  "requirement_candidate", "aspect_ref", "review_notes"}
    assert origin.sent_id.is_unique
    assert all(origin.loc[origin.rank_aleatoria.le(2), "sent_id"].isin(
        rankings.loc[rankings.ranking.eq("aleatoria"), "sent_id"]))
    selected_top = set(rankings.loc[rankings.ranking.ne("aleatoria") & rankings["rank"].le(2), "sent_id"])
    random_only = origin.loc[origin.selection_source.eq("aleatoria"), "sent_id"]
    assert len(random_only) == 2 and not set(random_only) & selected_top


def test_precision_at_k_requires_human_labels_and_counts_only_yes():
    origin = pd.DataFrame([
        {"review_id": 1, "rank_topico": 1},
        {"review_id": 2, "rank_topico": 2},
    ])
    annotations = pd.DataFrame({"review_id": [1, 2],
                                "requirement_candidate": ["sim", "nao"]})
    result = review.precision_at_k(annotations, origin, "topico", k=2)
    assert result["positives"] == 1 and result["precision"] == .5
    with pytest.raises(ValueError, match="anotacao"):
        review.precision_at_k(annotations.iloc[:1], origin, "topico", k=2)


def test_score_annotated_sample_reports_rank_precision_and_topic_counts():
    origin = pd.DataFrame([
        {"review_id": 1, "selection_source": "topico", "rank_topico": 1,
         "rank_lexical": 2, "rank_topico_lexical": 2, "global_topic_id": 4,
         "local_topic_id": 0},
        {"review_id": 2, "selection_source": "lexical", "rank_topico": 2,
         "rank_lexical": 1, "rank_topico_lexical": 1, "global_topic_id": 10,
         "local_topic_id": 0},
        {"review_id": 3, "selection_source": "aleatoria", "rank_topico": 3,
         "rank_lexical": 3, "rank_topico_lexical": 3, "global_topic_id": 4,
         "local_topic_id": 0},
    ])
    annotations = pd.DataFrame({"review_id": [1, 2, 3],
                                "requirement_candidate": ["sim", "nao", "sim"]})
    result, by_topic = review.score_annotated_sample(
        annotations, origin, top_k=1, n_random=1
    )
    assert result["topico"]["precision"] == 1.0
    assert result["lexical"]["precision"] == 0.0
    assert result["aleatoria"]["precision"] == 1.0
    assert by_topic.set_index("global_topic_id").loc[4, "n_sim"] == 2
