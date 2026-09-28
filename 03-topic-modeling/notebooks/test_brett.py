"""Regression checks for the BRETT feasibility gate and R bridge."""

import numpy as np
import pandas as pd
import pytest
from gensim.corpora import Dictionary

from _brett import (
    _run_brett_r,
    agrupar_categorias_raras,
    anchor_coverage_bound,
    build_covariate_design,
    build_tdm,
    fracao_documentos_vazios,
)


def test_tdm_uses_dictionary_order_and_counts():
    docs = [["battery", "charge", "battery"], ["screen", "screen"]]
    dictionary = Dictionary(docs)
    matrix, words = build_tdm(docs, dictionary)
    assert matrix.shape == (len(dictionary), len(docs))
    assert words[dictionary.token2id["battery"]] == "battery"
    assert matrix[dictionary.token2id["battery"], :].tolist() == [2, 0]


def test_category_grouping_preserves_input():
    categories = pd.Series(["laptop", "phone", "tablet"])
    grouped = agrupar_categorias_raras(
        categories, {"laptop": 500, "phone": 300, "tablet": 5}, 20
    )
    assert grouped.tolist() == ["laptop", "phone", "outras"]
    assert categories.tolist() == ["laptop", "phone", "tablet"]


def test_design_uses_most_common_reference_and_explicit_reference():
    categories = pd.Series(["laptop", "phone", "phone"])
    design = build_covariate_design(categories)
    assert design.columns.tolist() == ["Intercept", "cat_laptop"]
    assert design["cat_laptop"].tolist() == [1, 0, 0]
    other = build_covariate_design(categories, referencia="laptop")
    assert other.columns.tolist() == ["Intercept", "cat_phone"]
    assert (other["Intercept"] == 1).all()


def test_anchor_bound_proves_minimum_empty_fraction_for_any_topics():
    docs = [["a", "b"], ["a"], ["b"], ["c"], ["d"]]
    dictionary = Dictionary(docs)
    # For any one anchor, document coverage cannot exceed its DF (2/5).
    assert anchor_coverage_bound(dictionary, 5, topics=1) == pytest.approx(0.6)
    # Two anchors can cover at most 4/5 documents by the union bound.
    assert anchor_coverage_bound(dictionary, 5, topics=2) == pytest.approx(0.2)


def test_empty_fraction_counts_zero_theta_columns():
    theta = np.array([[1.0, 0.0, 0.2], [0.0, 0.0, 0.1]])
    assert fracao_documentos_vazios(theta) == pytest.approx(1 / 3)


def test_empty_fraction_rejects_missing_documents():
    with pytest.raises(ValueError):
        fracao_documentos_vazios(np.zeros((2, 0)))


def test_r_invoker_reports_missing_rscript(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Rscript"):
        _run_brett_r(tmp_path / "tdm.csv", tmp_path / "vocab.csv", tmp_path,
                     {"rscript_path": "missing-Rscript"}, seed=42, topics=2)
