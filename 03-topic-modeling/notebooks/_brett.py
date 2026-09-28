"""BRETT input preparation and anchor coverage checks.

The factorization and regression engine is the R package NMFregress. Its
anchor rows determine whether each sentence can receive a topic weight.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary


_BRETT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_brett.R"


def build_tdm(
    tokenized: list[list[str]], dictionary: Dictionary
) -> tuple[np.ndarray, list[str]]:
    """Build a dense word by document count matrix in Dictionary ID order."""
    matrix = np.zeros((len(dictionary), len(tokenized)), dtype=np.float64)
    for column, words in enumerate(tokenized):
        for token_id, count in dictionary.doc2bow(words):
            matrix[token_id, column] = count
    vocabulary = [dictionary[index] for index in range(len(dictionary))]
    return matrix, vocabulary


def agrupar_categorias_raras(
    categorias: pd.Series, videos_por_categoria: dict[str, int], min_videos: int
) -> pd.Series:
    """Group categories with fewer than ``min_videos`` distinct videos."""
    if min_videos < 1:
        raise ValueError("min_videos deve ser positivo")
    rare = {category for category, size in videos_por_categoria.items()
            if size < min_videos}
    return categorias.map(lambda category: "outras" if category in rare else category)


def build_covariate_design(
    categorias: pd.Series, referencia: str | None = None
) -> pd.DataFrame:
    """Return the explicit intercept and category dummies for NMFregress."""
    if categorias.empty or categorias.isna().any():
        raise ValueError("Categorias devem ser nao vazias e sem valores ausentes")
    if referencia is None:
        referencia = str(categorias.value_counts().idxmax())
    if referencia not in categorias.values:
        raise ValueError("Categoria de referencia ausente")
    design = pd.DataFrame({"Intercept": np.ones(len(categorias), dtype=float)})
    for category in sorted(categorias.unique()):
        if category != referencia:
            design[f"cat_{category}"] = (categorias == category).to_numpy(dtype=float)
    return design


def anchor_coverage_bound(dictionary: Dictionary, n_docs: int, topics: int) -> float:
    """Lower bound on empty documents for *any* selection of ``topics`` anchors.

    The covered document set is a union of at most ``topics`` token presence
    sets. Its size cannot exceed the sum of the largest document frequencies.
    Thus this bound can reject an infeasible topic count before allocating a
    corpus-sized dense term-document matrix or fitting BRETT.
    """
    if n_docs < 1 or topics < 1:
        raise ValueError("n_docs e topics devem ser positivos")
    max_covered = sum(sorted(dictionary.dfs.values(), reverse=True)[:topics])
    return float(max(0.0, 1.0 - max_covered / n_docs))


def fracao_documentos_vazios(theta: np.ndarray, tol: float = 1e-8) -> float:
    """Fraction of document columns with no topic weight (theta is T by D)."""
    values = np.asarray(theta, dtype=float)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or tol < 0:
        raise ValueError("theta deve ser uma matriz T x D nao vazia")
    return float(np.all(np.abs(values) <= tol, axis=0).mean())


def _run_brett_r(
    tdm_csv: Path, vocab_csv: Path, output_dir: Path,
    brett_cfg: dict, seed: int, topics: int, covariates_csv: Path | None = None,
    timeout_sec: int = 21600,
) -> dict:
    """Run the official NMFregress factorization through the local R bridge."""
    rscript = brett_cfg.get("rscript_path", "Rscript")
    if shutil.which(rscript) is None and not Path(rscript).exists():
        raise RuntimeError(f"Rscript nao encontrado: {rscript}")
    if not _BRETT_SCRIPT.is_file():
        raise RuntimeError(f"Script R nao encontrado: {_BRETT_SCRIPT}")
    mode = brett_cfg.get("mode", "fit")
    if mode not in {"fit", "fit_and_regress"}:
        raise ValueError(f"Modo BRETT desconhecido: {mode}")
    command = [rscript, str(_BRETT_SCRIPT), "--mode", mode,
               "--tdm", str(tdm_csv), "--vocab", str(vocab_csv),
               "--output_dir", str(output_dir), "--seed", str(seed),
               "--topics", str(topics), "--bootstrap_n",
               str(brett_cfg.get("bootstrap_n", 1000))]
    if covariates_csv is not None:
        command.extend(["--covariates", str(covariates_csv)])
    env = os.environ.copy()
    local_lib = brett_cfg.get("r_lib", _BRETT_SCRIPT.parent.parent / "data/output/.r_lib")
    env["R_LIBS"] = str(local_lib)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec,
                            encoding="utf-8", errors="replace", env=env)
    if result.returncode:
        raise RuntimeError(f"run_brett.R exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    final = Path(output_dir) / "brett_final.json"
    if not final.is_file():
        raise RuntimeError(f"R nao gravou {final}\n{result.stdout}\n{result.stderr}")
    return json.loads(final.read_text(encoding="utf-8"))
