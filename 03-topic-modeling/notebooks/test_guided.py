"""Contracts for the seeded topic models (H6)."""

import numpy as np
import pytest
from gensim.corpora import Dictionary
from scipy import sparse

from _guided import (
    apply_seed_exclusion_rule,
    build_seed_matrix,
    coefficient_of_variation,
    escolher_lambda,
    guided_nmf,
    load_seeds,
    purity_by_group,
    topic_concordance,
    _run_keyatm_r,
    category_dependence,
    bootstrap_prevalence_by_video,
    seeded_topic_stability,
    seed_topic_assignment,
    normalize_seeds,
)


def test_seed_file_and_exclusion_record_frequency(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("aspectos:\n  bateria: [battery, charge, mah]\n  tela: [screen]\n", encoding="utf-8")
    seeds = load_seeds(path)
    dictionary = Dictionary([["battery", "screen"], ["battery", "charge"]])
    effective, log = apply_seed_exclusion_rule(seeds, dictionary, min_df=2)
    assert effective == {}
    assert set(log["sementes_excluidas"]) == {"charge", "mah", "screen"}
    assert log["aspectos_excluidos"] == ["bateria", "tela"]
    assert log["frequencias"]["battery"] == 2
    assert log["frequencias"]["mah"] == 0


def test_seed_file_requires_aspectos(tmp_path):
    path = tmp_path / "seeds.yaml"
    path.write_text("other: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aspectos"):
        load_seeds(path)


def test_normalize_seeds_uses_corpus_lemmas_and_extra_stopwords():
    normalized = normalize_seeds(
        {"bateria": ["batteries", "battery", "charging", "charge", "yeah"]},
        extra_stopwords=["yeah"],
    )
    assert normalized == {"bateria": ["battery", "charge"]}


def test_seed_matrix_aligns_dictionary_ids_and_empty_case():
    dictionary = Dictionary([["battery", "screen", "keyboard"]])
    matrix, names = build_seed_matrix({"bateria": ["battery"], "tela": ["screen"]}, dictionary)
    assert names == ["bateria", "tela"]
    assert matrix.shape == (3, 2)
    assert matrix[dictionary.token2id["battery"], 0] == 1
    assert matrix[dictionary.token2id["screen"], 0] == 0
    empty, empty_names = build_seed_matrix({}, dictionary)
    assert empty.shape == (3, 0)
    assert empty_names == []


def test_category_and_topic_measures():
    assert coefficient_of_variation([1, 2, 3]) == pytest.approx(np.std([1, 2, 3]) / 2)
    assert np.isnan(coefficient_of_variation([0, 0]))
    labels = np.array([0, 0, 1, 1])
    assert topic_concordance(labels, labels) == pytest.approx({"nmi": 1, "ari": 1})
    purity = purity_by_group(np.array([2, 2, 3, 4]), labels)
    assert purity.loc[purity["grupo"] == 0, "fracao_majoritaria"].iloc[0] == 1
    assert purity.loc[purity["grupo"] == 1, "fracao_majoritaria"].iloc[0] == 0.5


def test_purity_restricts_match_to_seeded_topics():
    purity = purity_by_group(np.array([99, 99, 99, 1]), np.array([0, 0, 0, 0]),
                             allowed_topics=[1, 2])
    assert purity.iloc[0]["topico_majoritario"] == 1
    assert purity.iloc[0]["fracao_majoritaria"] == pytest.approx(0.25)


def test_guided_nmf_lambda_zero_has_ordinary_nmf_updates_and_sparse_input():
    x = sparse.csr_matrix(np.array([[3., 0., 1.], [0., 2., 0.], [1., 0., 4.]]))
    y = np.array([[1.], [0.], [0.]])
    with_seed = guided_nmf(x, y, k=2, lam=0, seed=42, max_iter=30)
    without_seed = guided_nmf(x, np.zeros((3, 0)), k=2, lam=0, seed=42, max_iter=30)
    np.testing.assert_allclose(with_seed["A"], without_seed["A"], rtol=1e-12)
    np.testing.assert_allclose(with_seed["S"], without_seed["S"], rtol=1e-12)
    assert all(b <= a + 1e-8 for a, b in zip(with_seed["loss_history"], with_seed["loss_history"][1:]))


def test_guided_nmf_recovers_seeded_words_on_synthetic_corpus():
    rng = np.random.default_rng(7)
    a_true = rng.random((30, 3)) * 0.1
    a_true[:5, 0] = 1
    x = a_true @ rng.random((3, 200)) + rng.random((30, 200)) * 0.01
    y = np.zeros((30, 1))
    y[:5, 0] = 1
    result = guided_nmf(x, y, k=3, lam=5, seed=7, max_iter=500)
    topic = np.argmax(result["B"][:, 0])
    assert set(np.argsort(-result["A"][:, topic])[:5]) == set(range(5))


def test_choose_lambda_uses_smallest_threshold_hit():
    x = np.array([[4., 4., 4.], [3., 3., 3.], [0., 1., 0.], [0., 0., 1.]])
    y = np.array([[1.], [1.], [0.], [0.]])
    chosen = escolher_lambda(x, y, k=2, seed=3, grid=[0.5, 0.1, 1], top_n=2, frac_min=1)
    assert chosen == 0.1


def test_choose_lambda_warns_if_grid_never_reaches_seed_fraction():
    x = np.array([[2., 2.], [1., 1.], [0., 1.]])
    y = np.array([[1.], [1.], [0.]])
    with pytest.warns(RuntimeWarning, match="nenhum lambda"):
        chosen = escolher_lambda(x, y, k=2, seed=1, grid=[0.1, 0.5],
                                top_n=1, frac_min=1)
    assert chosen == 0.5


def test_run_keyatm_reports_missing_rscript(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Rscript"):
        _run_keyatm_r("base", tmp_path / "input.csv", tmp_path / "keywords.json",
                      tmp_path, {"rscript_path": "not-installed-Rscript"}, seed=42)


def test_run_keyatm_requires_covariates_in_covariate_mode(tmp_path):
    with pytest.raises(ValueError, match="covariates"):
        _run_keyatm_r("covariates", tmp_path / "input.csv", tmp_path / "keywords.json",
                      tmp_path, {}, seed=42)


def test_run_keyatm_uses_project_local_r_library(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from _guided import _KEYATM_SCRIPT

    local_lib = tmp_path / "r_lib"
    local_lib.mkdir()
    monkeypatch.setattr("shutil.which", lambda _: "Rscript")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        (tmp_path / "keyatm_final.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)
    assert _KEYATM_SCRIPT.exists()
    _run_keyatm_r("base", tmp_path / "input.csv", tmp_path / "keywords.json",
                  tmp_path, {"r_lib": str(local_lib), "iterations": 50}, seed=42)
    assert captured["env"]["R_LIBS"] == str(local_lib)
    assert captured["env"].get("R_LIBS_USER") is None
    assert captured["command"][-2:] == ["--iterations", "50"]


def test_category_dependence_groups_rare_categories_by_video_count():
    theta = np.array([[0.2], [0.4], [0.6], [0.8]])
    categories = ["laptop", "laptop", "phone", "tablet"]
    videos = ["v1", "v1", "v2", "v3"]
    table = category_dependence(theta, categories, videos, min_videos=2)
    assert set(table["categoria"]) == {"outras"}
    assert table.iloc[0]["n_videos"] == 3
    assert table.iloc[0]["prevalencia"] == pytest.approx(0.5)


def test_category_dependence_can_attach_video_bootstrap_intervals():
    theta = np.array([[0.], [0.], [1.], [1.]])
    table = category_dependence(theta, ["phone"] * 4, ["a", "a", "b", "b"],
                                min_videos=1, n_boot=100, seed=0)
    assert table.iloc[0]["ci_low"] == pytest.approx(0.0)
    assert table.iloc[0]["ci_high"] == pytest.approx(1.0)


def test_bootstrap_reuses_all_sentences_in_a_sampled_video():
    theta = np.array([[0.], [0.], [1.]])
    table = bootstrap_prevalence_by_video(theta, ["a", "a", "b"], n_boot=500, seed=3)
    # Sampling individual sentences would center at 1/3; equal video weighting
    # while preserving clusters yields the two extremes and the original 1/3.
    assert table.iloc[0]["ci_low"] == pytest.approx(0.0)
    assert table.iloc[0]["ci_high"] == pytest.approx(1.0)
    assert table.iloc[0]["prevalencia"] == pytest.approx(1 / 3)


def test_seeded_stability_matches_aspect_names_not_topic_position():
    runs = [
        {"bateria": ["battery", "charge"], "tela": ["screen", "display"]},
        {"tela": ["display", "screen"], "bateria": ["charge", "battery"]},
    ]
    table = seeded_topic_stability(runs)
    assert set(table["aspecto"]) == {"bateria", "tela"}
    assert table["jaccard"].tolist() == [1.0, 1.0]


def test_seed_topic_assignment_keeps_aspect_topics_distinct():
    weights = np.array([[0.9, 0.8], [0.1, 0.7]])
    assert seed_topic_assignment(weights) == [0, 1]
