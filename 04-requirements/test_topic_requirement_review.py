import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_requirement_review import join_topic_evidence  # noqa: E402
import topic_requirement_review as review  # noqa: E402


def test_join_topic_evidence_links_sentence_stm_and_nmf_without_losing_rows():
    sentences = pd.DataFrame([
        {"sent_id": "video_a_0", "post_id": "video_a", "sentence": "One port is not enough.",
         "context": "One port is not enough.", "product_category": "laptop", "source": "whisper"},
        {"sent_id": "video_b_0", "post_id": "video_b", "sentence": "The fan is noisy.",
         "context": "The fan is noisy.", "product_category": "laptop", "source": "caption"},
    ])
    stm = pd.DataFrame([
        {"post_id": "video_a_0", "topic_id": 10, "topic_name": "Ports",
         "topic_prob_distribution": "[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9]"},
        {"post_id": "video_b_0", "topic_id": 4, "topic_name": "Cooling",
         "topic_prob_distribution": "[0.01, 0.01, 0.01, 0.01, 0.8]"},
    ])
    nmf = {
        10: pd.DataFrame([{"post_id": "video_a_0", "topic_id": 1,
                           "topic_name": "USB layout",
                           "topic_prob_distribution": "[0.3, 0.7]"}]),
        4: pd.DataFrame([{"post_id": "video_b_0", "topic_id": 0,
                          "topic_name": "Fan noise",
                          "topic_prob_distribution": "[0.8, 0.2]"}]),
    }
    out = join_topic_evidence(sentences, stm, nmf, expected_topics=[4, 10])
    assert out.sent_id.tolist() == ["video_a_0", "video_b_0"]
    assert out.stm_topic_id.tolist() == [10, 4]
    assert out.nmf_topic_id.tolist() == [1, 0]
    assert out.nmf_topic_weight.tolist() == pytest.approx([0.7, 0.8])
    assert out.stm_topic_weight.tolist() == pytest.approx([0.9, 0.8])
    assert out.post_id.tolist() == ["video_a", "video_b"]
    assert out.stm_topic_name.tolist() == ["Ports", "Cooling"]
    assert out.nmf_topic_name.tolist() == ["USB layout", "Fan noise"]


def test_join_topic_evidence_rejects_missing_run_or_sentence():
    sentences = pd.DataFrame([{"sent_id": "video_a_0", "post_id": "video_a",
                               "sentence": "A", "context": "A", "product_category": "phone"}])
    stm = pd.DataFrame([{"post_id": "video_a_0", "topic_id": 10}])
    nmf = {10: pd.DataFrame([{"post_id": "absent_0", "topic_id": 0,
                              "topic_prob_distribution": "[1.0]"}])}
    with pytest.raises(ValueError, match="faltando"):
        join_topic_evidence(sentences, stm, nmf, expected_topics=[4, 10])
    with pytest.raises(ValueError, match="sent_id"):
        join_topic_evidence(sentences, stm, nmf, expected_topics=[10])


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
        {"sent_id": "a", "nmf_topic_weight": .9, "lex_hit": False, "lex_n": 0},
        {"sent_id": "b", "nmf_topic_weight": .6, "lex_hit": True, "lex_n": 1},
        {"sent_id": "c", "nmf_topic_weight": .7, "lex_hit": True, "lex_n": 1},
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
                             "nmf_topic_weight": [1.0] * 8,
                             "lex_hit": [False] * 8, "lex_n": [0] * 8})
    order = review.build_rankings(evidence, seed=42)
    ranked = order.loc[order.ranking.eq("topico"), "sent_id"].tolist()
    assert ranked != list("abcdefgh")
    assert ranked == review.build_rankings(evidence, seed=42).loc[
        lambda frame: frame.ranking.eq("topico"), "sent_id"
    ].tolist()


def test_topic_ranking_uses_stm_and_nmf_confidence_together():
    evidence = pd.DataFrame([
        {"sent_id": "a", "nmf_topic_weight": .9, "stm_topic_weight": .1,
         "lex_hit": False, "lex_n": 0},
        {"sent_id": "b", "nmf_topic_weight": .6, "stm_topic_weight": .9,
         "lex_hit": False, "lex_n": 0},
    ])
    ranked = review.build_rankings(evidence, seed=42)
    assert ranked.loc[ranked.ranking.eq("topico"), "sent_id"].iloc[0] == "b"


def test_blind_sample_keeps_origin_separate_and_random_outside_top_lists():
    evidence = pd.DataFrame([
        {"sent_id": f"v_{i}", "sentence": f"Sentence {i}", "context": f"Context {i}",
         "product_category": "laptop", "stm_topic_id": 10, "nmf_topic_id": 0,
         "nmf_topic_weight": 1 - i / 10, "lex_hit": i in {2, 3},
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


def test_discover_nmf_results_requires_one_run_per_aspect_topic(tmp_path):
    folder = tmp_path / "topic04" / "topic04_20260922_010000"
    folder.mkdir(parents=True)
    (folder / "nmf_results.csv").write_text("post_id,topic_id\n", encoding="utf-8")
    assert review.discover_nmf_results(tmp_path, [4]) == {4: folder / "nmf_results.csv"}
    with pytest.raises(ValueError, match="topic10"):
        review.discover_nmf_results(tmp_path, [4, 10])


def test_run_review_exports_joined_evidence_rankings_and_blind_sample(tmp_path):
    sentences = pd.DataFrame([
        {"sent_id": f"video_{i}", "post_id": "video", "sentence":
         "I wish it had a port." if i % 2 else "It has a port.",
         "context": f"Context {i}", "product_category": "laptop", "source": "whisper"}
        for i in range(8)
    ])
    stm = pd.DataFrame([{"post_id": f"video_{i}", "topic_id": 4 if i < 4 else 10,
                         "topic_prob_distribution": "[0.01, 0.01, 0.01, 0.01, 0.8]"
                         if i < 4 else "[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.8]"}
                        for i in range(8)])
    nmf = {topic: pd.DataFrame([{"post_id": f"video_{i}", "topic_id": 0,
                                 "topic_prob_distribution": "[1.0]"}
                                for i in indices])
           for topic, indices in [(4, range(4)), (10, range(4, 8))]}
    summary = review.run_review(sentences, stm, nmf, [4, 10], tmp_path,
                                top_k=1, n_random=2, seed=42)
    assert summary["n_evidence"] == 8
    assert summary["n_lex_hit"] == 4
    evidence = pd.read_csv(tmp_path / "evidencias.csv")
    assert evidence.sent_id.is_unique and len(evidence) == 8
    assert evidence.stm_topic_weight.notna().all()
    assert len(pd.read_csv(tmp_path / "rankings.csv")) == 32
    blind = pd.read_csv(tmp_path / "amostra_cega.csv")
    assert blind.requirement_candidate.isna().all()
    assert (tmp_path / "amostra_origem.csv").is_file()
    assert (tmp_path / "metadata.json").is_file()


def test_score_annotated_sample_reports_rank_precision_and_topic_counts():
    origin = pd.DataFrame([
        {"review_id": 1, "selection_source": "topico", "rank_topico": 1,
         "rank_lexical": 2, "rank_topico_lexical": 2, "stm_topic_id": 4},
        {"review_id": 2, "selection_source": "lexical", "rank_topico": 2,
         "rank_lexical": 1, "rank_topico_lexical": 1, "stm_topic_id": 10},
        {"review_id": 3, "selection_source": "aleatoria", "rank_topico": 3,
         "rank_lexical": 3, "rank_topico_lexical": 3, "stm_topic_id": 4},
    ])
    annotations = pd.DataFrame({"review_id": [1, 2, 3],
                                "requirement_candidate": ["sim", "nao", "sim"]})
    result, by_topic = review.score_annotated_sample(
        annotations, origin, top_k=1, n_random=1
    )
    assert result["topico"]["precision"] == 1.0
    assert result["lexical"]["precision"] == 0.0
    assert result["aleatoria"]["precision"] == 1.0
    assert by_topic.set_index("stm_topic_id").loc[4, "n_sim"] == 2
