"""Seeded topic models for H6; seeds come only from configs/seeds_aspectos.yaml."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


_KEYATM_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_keyatm.R"


def load_seeds(seeds_path: str | Path) -> dict[str, list[str]]:
    """Read the frozen aspect lexicon, preserving its declared aspect order."""
    with open(seeds_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("aspectos"), dict):
        raise ValueError(f"YAML de sementes sem a chave 'aspectos': {seeds_path}")
    seeds = data["aspectos"]
    if any(not isinstance(words, list) or not all(isinstance(word, str) for word in words)
           for words in seeds.values()):
        raise ValueError("Cada aspecto deve conter uma lista de sementes textuais")
    return seeds


def normalize_seeds(
    seeds: dict[str, list[str]], lang: str = "en",
    extra_stopwords: list[str] | None = None,
    preserve_terms: list[str] | None = None,
) -> dict[str, list[str]]:
    """Pass seeds through the same spaCy lemmatizer and stopword rule as STM."""
    from _helpers import _get_nlp

    nlp = _get_nlp(lang)
    extra = {word.lower() for word in (extra_stopwords or [])}
    preserved = {word.lower() for word in (preserve_terms or [])}
    normalized = {}
    for aspect, words in seeds.items():
        effective = []
        for word, doc in zip(words, nlp.pipe(words)):
            literal = word.lower()
            if literal in preserved:
                tokens = [literal] if literal.isalpha() and len(literal) > 2 and literal not in extra else []
            else:
                tokens = [token.lemma_.lower() for token in doc
                          if not token.is_stop and token.is_alpha
                          and len(token.lemma_) > 2 and token.lemma_.lower() not in extra]
            if len(tokens) == 1 and tokens[0] not in effective:
                effective.append(tokens[0])
        normalized[aspect] = effective
    return normalized


def apply_seed_exclusion_rule(
    seeds: dict[str, list[str]],
    dictionary: Dictionary,
    min_df: int,
    min_sementes_por_aspecto: int = 2,
) -> tuple[dict[str, list[str]], dict]:
    """Exclude absent or rare seeds and aspects with too few surviving seeds."""
    if min_df < 1 or min_sementes_por_aspecto < 1:
        raise ValueError("Os limiares de exclusao devem ser positivos")
    effective: dict[str, list[str]] = {}
    removed_seeds: list[str] = []
    removed_aspects: list[str] = []
    frequencies: dict[str, int] = {}
    for aspect, words in seeds.items():
        kept: list[str] = []
        for word in dict.fromkeys(words):
            token_id = dictionary.token2id.get(word)
            frequency = int(dictionary.dfs.get(token_id, 0)) if token_id is not None else 0
            frequencies[word] = frequency
            if frequency < min_df:
                removed_seeds.append(word)
            else:
                kept.append(word)
        if len(kept) < min_sementes_por_aspecto:
            removed_aspects.append(aspect)
        else:
            effective[aspect] = kept
    return effective, {
        "sementes_excluidas": removed_seeds,
        "aspectos_excluidos": removed_aspects,
        "frequencias": frequencies,
    }


def build_seed_matrix(
    seeds_efetivas: dict[str, list[str]], dictionary: Dictionary
) -> tuple[np.ndarray, list[str]]:
    """Build vocabulary by aspect matrix in Dictionary token-id order."""
    names = list(seeds_efetivas)
    matrix = np.zeros((len(dictionary), len(names)), dtype=float)
    for column, aspect in enumerate(names):
        for word in seeds_efetivas[aspect]:
            matrix[dictionary.token2id[word], column] = 1.0
    return matrix, names


def scaled_lambda_grid(X, Y, alpha_grid):
    """Map dimensionless supervision weights to the corpus-specific objective."""
    y = np.asarray(Y, dtype=float)
    x_norm = float(X.multiply(X).sum()) if sparse.issparse(X) else float(np.sum(np.asarray(X) ** 2))
    y_norm = float(np.sum(y ** 2))
    if y_norm <= 0:
        raise ValueError("matriz de semente vazia")
    if x_norm <= 0 or not alpha_grid or any(not np.isfinite(a) or a <= 0 for a in alpha_grid):
        raise ValueError("X ou grade alpha invalida")
    return [float(alpha * x_norm / y_norm) for alpha in alpha_grid]


def coefficient_of_variation(valores: list[float] | np.ndarray) -> float:
    values = np.asarray(valores, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    mean = values.mean()
    return float(values.std(ddof=0) / mean) if mean != 0 else float("nan")


def topic_concordance(atribuicao_a: np.ndarray, atribuicao_b: np.ndarray) -> dict[str, float]:
    if len(atribuicao_a) != len(atribuicao_b):
        raise ValueError("As atribuicoes devem ter o mesmo numero de linhas")
    return {
        "nmi": float(normalized_mutual_info_score(atribuicao_a, atribuicao_b)),
        "ari": float(adjusted_rand_score(atribuicao_a, atribuicao_b)),
    }


def purity_by_group(
    atribuicao_guiada: np.ndarray, grupo_referencia: np.ndarray,
    allowed_topics: list[int] | None = None,
) -> pd.DataFrame:
    if len(atribuicao_guiada) != len(grupo_referencia):
        raise ValueError("As atribuicoes devem ter o mesmo numero de linhas")
    counts = pd.DataFrame({"grupo": grupo_referencia, "topico": atribuicao_guiada})
    if allowed_topics is not None and not allowed_topics:
        raise ValueError("allowed_topics nao pode ser vazio")
    rows = []
    for group, sub in counts.groupby("grupo", sort=True):
        frequency = sub["topico"].value_counts()
        if allowed_topics is not None:
            frequency = frequency.reindex(allowed_topics, fill_value=0)
        rows.append({"grupo": group, "topico_majoritario": frequency.idxmax(),
                     "fracao_majoritaria": float(frequency.max() / len(sub)), "n": len(sub)})
    return pd.DataFrame(rows, columns=["grupo", "topico_majoritario", "fracao_majoritaria", "n"])


def category_dependence(
    theta: np.ndarray, categories: list[str], video_ids: list[str], min_videos: int = 20,
    n_boot: int = 0, seed: int = 42,
) -> pd.DataFrame:
    """Mean topic prevalence by category, grouping categories with few videos."""
    values = np.asarray(theta, dtype=float)
    if values.ndim != 2 or len(categories) != len(values) or len(video_ids) != len(values):
        raise ValueError("theta, categorias e videos devem ter linhas alinhadas")
    if min_videos < 1 or n_boot < 0:
        raise ValueError("min_videos deve ser positivo e n_boot nao negativo")
    frame = pd.DataFrame({"categoria": categories, "video": video_ids})
    sizes = frame.groupby("categoria")["video"].nunique()
    frame.loc[frame["categoria"].map(sizes) < min_videos, "categoria"] = "outras"
    rows = []
    means = frame.groupby("categoria", sort=True).indices
    intervals = {}
    if n_boot:
        for category, indices in means.items():
            intervals[category] = bootstrap_prevalence_by_video(
                values[indices], frame.iloc[indices]["video"].tolist(),
                n_boot=n_boot, seed=seed,
            )
    for topic in range(values.shape[1]):
        topic_means = {category: float(values[indices, topic].mean())
                       for category, indices in means.items()}
        cv = coefficient_of_variation(list(topic_means.values()))
        phone = topic_means.get("phone", float("nan"))
        ratio = topic_means.get("laptop", float("nan")) / phone if phone > 0 else float("nan")
        for category, indices in means.items():
            interval = intervals.get(category)
            rows.append({"topico": topic, "categoria": category,
                         "prevalencia": topic_means[category],
                         "n_videos": frame.iloc[indices]["video"].nunique(),
                         "n_docs": len(indices), "cv": cv,
                         "razao_laptop_phone": ratio,
                         "ci_low": float(interval.iloc[topic]["ci_low"]) if interval is not None else float("nan"),
                         "ci_high": float(interval.iloc[topic]["ci_high"]) if interval is not None else float("nan")})
    return pd.DataFrame(rows)


def bootstrap_prevalence_by_video(
    theta: np.ndarray, video_ids: list[str], n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Cluster bootstrap: draw videos, retaining every sentence in each draw."""
    values = np.asarray(theta, dtype=float)
    if values.ndim != 2 or len(video_ids) != len(values) or not len(values) or n_boot < 1:
        raise ValueError("theta e video_ids desalinhados ou n_boot invalido")
    ids = pd.Index(video_ids)
    unique = ids.unique().to_numpy()
    blocks = [np.flatnonzero(ids == video_id) for video_id in unique]
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, values.shape[1]))
    for iteration in range(n_boot):
        sampled = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in sampled])
        means[iteration] = values[indices].mean(axis=0)
    return pd.DataFrame({"topico": np.arange(values.shape[1]),
                         "prevalencia": values.mean(axis=0),
                         "ci_low": np.quantile(means, 0.025, axis=0),
                         "ci_high": np.quantile(means, 0.975, axis=0)})


def seeded_topic_stability(runs: list[dict[str, list[str]]]) -> pd.DataFrame:
    """Pairwise Jaccard of seeded topics by aspect name, regardless of run order."""
    if len(runs) < 2:
        raise ValueError("Pelo menos duas seeds de treinamento sao necessarias")
    aspects = sorted(set.intersection(*(set(run) for run in runs)))
    rows = []
    for aspect in aspects:
        for first in range(len(runs)):
            for second in range(first + 1, len(runs)):
                left, right = set(runs[first][aspect][:10]), set(runs[second][aspect][:10])
                rows.append({"aspecto": aspect, "run_a": first, "run_b": second,
                             "jaccard": len(left & right) / len(left | right)
                             if left or right else 1.0})
    return pd.DataFrame(rows)


def seed_topic_assignment(weights: np.ndarray) -> list[int]:
    """Assign one distinct learned topic to each aspect by its B weight."""
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("B deve ser topicos x aspectos, com k >= aspectos")
    aspects, topics = linear_sum_assignment(-matrix.T)
    assignment = np.empty(matrix.shape[1], dtype=int)
    assignment[aspects] = topics
    return assignment.tolist()


def guided_nmf(
    X: np.ndarray | sparse.spmatrix,
    Y: np.ndarray,
    k: int,
    lam: float,
    seed: int,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict:
    """Minimize ||X-AS||² + lam||Y-AB||² with multiplicative updates.

    X is vocabulary by document and may be sparse. The objective is evaluated
    through small Gram matrices, so a corpus-sized dense matrix is never made.
    """
    x = sparse.csr_matrix(X, dtype=float) if sparse.issparse(X) else np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("X e Y devem ser matrizes com o mesmo vocabulario")
    x_values = x.data if sparse.issparse(x) else x
    if k < 1 or lam < 0 or max_iter < 1 or np.any(x_values < 0) or np.any(y < 0):
        raise ValueError("k e max_iter devem ser positivos; X, Y e lambda nao negativos")
    if y.shape[1] > k:
        raise ValueError("k deve comportar todos os aspectos semeados")
    m, n = x.shape
    c = y.shape[1]
    rng = np.random.default_rng(seed)
    eps = 1e-12
    a = rng.random((m, k)) + eps
    s = rng.random((k, n)) + eps
    b = rng.random((k, c)) + eps
    x_norm_sq = float(x.multiply(x).sum()) if sparse.issparse(x) else float(np.sum(x * x))
    y_norm_sq = float(np.sum(y * y))

    def loss() -> float:
        atx = np.asarray(a.T @ x)
        gram_a = a.T @ a
        gram_s = s @ s.T
        reconstruction = x_norm_sq - 2 * np.sum(atx * s) + np.sum(gram_a * gram_s)
        seed_loss = y_norm_sq - 2 * np.sum((a.T @ y) * b) + np.sum(gram_a * (b @ b.T))
        return float(max(0.0, reconstruction) + lam * max(0.0, seed_loss))

    history = [loss()]
    for _ in range(max_iter):
        s *= np.asarray(a.T @ x) / (a.T @ a @ s + eps)
        if c and lam:
            b *= (a.T @ y) / (a.T @ a @ b + eps)
        numerator = np.asarray(x @ s.T)
        denominator = a @ (s @ s.T)
        if c and lam:
            numerator += lam * (y @ b.T)
            denominator += lam * (a @ (b @ b.T))
        a *= numerator / (denominator + eps)
        current = loss()
        history.append(current)
        if abs(history[-2] - current) <= tol * max(1.0, history[-2]):
            break
    return {"A": a, "S": s, "B": b, "loss_history": history}


def escolher_lambda(
    X: np.ndarray | sparse.spmatrix, Y: np.ndarray, k: int, seed: int,
    grid: list[float], top_n: int, frac_min: float,
    max_iter: int = 200, diagnostics: list[dict] | None = None,
) -> float:
    """Choose the smallest lambda meeting seed recall, without STM measures."""
    if not grid or any(value <= 0 for value in grid) or top_n < 1 or not 0 < frac_min <= 1:
        raise ValueError("Grade, top_n ou fracao de lambda invalidos")
    if Y.shape[1] == 0:
        raise ValueError("Nenhum aspecto semeado sobreviveu")
    for lam in sorted(set(grid)):
        result = guided_nmf(X, Y, k=k, lam=lam, seed=seed, max_iter=max_iter)
        a, b = result["A"], result["B"]
        fractions = []
        assigned_topics = seed_topic_assignment(b)
        for column in range(Y.shape[1]):
            topic = assigned_topics[column]
            top_words = set(np.argsort(-a[:, topic])[:top_n])
            seeds = set(np.flatnonzero(Y[:, column]))
            fractions.append(len(top_words & seeds) / len(seeds) if seeds else 0.0)
        if diagnostics is not None:
            diagnostics.append({"lambda": float(lam), "seed_recall": fractions,
                                "min_seed_recall": min(fractions), "max_iter": max_iter})
        if min(fractions) >= frac_min:
            return float(lam)
    warnings.warn(
        f"nenhum lambda da grade {sorted(set(grid))} atingiu a fracao minima {frac_min}; "
        f"usando {max(grid)} e registrando limitacao metodologica",
        RuntimeWarning,
        stacklevel=2,
    )
    return float(max(grid))


def _run_keyatm_r(
    mode: str, input_csv, keywords_json, output_dir, keyatm_cfg: dict,
    seed: int, covariates_csv=None, covariates_formula: str | None = None,
    timeout_sec: int = 21600,
) -> dict:
    """Run the R model and surface R failures with their original output."""
    if mode not in {"base", "covariates"}:
        raise ValueError(f"Modo keyATM desconhecido: {mode}")
    if mode == "covariates" and (covariates_csv is None or not covariates_formula):
        raise ValueError("Modo covariates exige CSV e formula")
    rscript = keyatm_cfg.get("rscript_path", "Rscript")
    if shutil.which(rscript) is None and not Path(rscript).exists():
        raise RuntimeError(f"Rscript nao encontrado ({rscript!r})")
    if not _KEYATM_SCRIPT.exists():
        raise RuntimeError(f"Script R nao encontrado: {_KEYATM_SCRIPT}")
    command = [rscript, str(_KEYATM_SCRIPT), "--mode", mode, "--input", str(input_csv),
               "--keywords", str(keywords_json), "--output_dir", str(output_dir),
               "--seed", str(seed), "--no_keyword_topics",
               str(keyatm_cfg.get("no_keyword_topics", 5))]
    if mode == "covariates":
        command.extend(["--covariates", str(covariates_csv),
                        "--covariates_formula", covariates_formula])
    if "iterations" in keyatm_cfg:
        command.extend(["--iterations", str(int(keyatm_cfg["iterations"]))])
    env = os.environ.copy()
    local_lib = keyatm_cfg.get("r_lib")
    if local_lib is None:
        candidate = _KEYATM_SCRIPT.parent.parent / "data" / "output" / ".r_lib"
        local_lib = candidate if candidate.exists() else None
    if local_lib is not None:
        library_path = Path(local_lib)
        if not library_path.is_absolute():
            library_path = _KEYATM_SCRIPT.parent.parent / library_path
        env["R_LIBS"] = str(library_path)
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec,
                            encoding="utf-8", errors="replace", env=env)
    if result.returncode:
        raise RuntimeError(f"run_keyatm.R exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    final_path = Path(output_dir) / "keyatm_final.json"
    if not final_path.exists():
        raise RuntimeError(f"R nao gravou {final_path}\n{result.stdout}\n{result.stderr}")
    with final_path.open(encoding="utf-8") as handle:
        return json.load(handle)
