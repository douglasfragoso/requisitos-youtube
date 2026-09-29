"""Run the predeclared Guided NMF context-window v3 experiment.

Each retained STM sentence is represented by itself and its immediate
neighbors from the same video. Seed lexicon, STM run, lambda rule and pass
criteria remain frozen from the previous round.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

ROOT = Path(__file__).resolve().parents[2]
TOPIC = ROOT / "03-topic-modeling"
sys.path.insert(0, str(TOPIC / "notebooks"))
from _brett import context_windows  # noqa: E402
from _guided import (  # noqa: E402
    apply_seed_exclusion_rule, build_seed_matrix, category_dependence,
    escolher_lambda, guided_nmf, load_seeds, normalize_seeds,
    purity_by_group, scaled_lambda_grid, seed_topic_assignment,
    topic_concordance,
)
from _helpers import make_run_output_dir  # noqa: E402


def main() -> int:
    params = yaml.safe_load((TOPIC / "configs/params.yaml").read_text(encoding="utf-8"))
    cfg = params["guided"]
    model_cfg = cfg["gnmf"]
    corpus_cfg = params["corpora"]["youtube_sent"]
    stm_dir = TOPIC / "data/output/youtube_sent/stm" / model_cfg["stm_run"]
    required = ["stm_input.csv", "stm_theta.csv", "stm_final.json", "stm_results.csv"]
    if not all((stm_dir / name).is_file() for name in required):
        raise FileNotFoundError(f"Run STM incompleto: {stm_dir}")

    original = pd.read_csv(stm_dir / "stm_input.csv")
    stm_meta = json.loads((stm_dir / "stm_final.json").read_text(encoding="utf-8"))
    theta = pd.read_csv(stm_dir / "stm_theta.csv").to_numpy()
    stm_result = pd.read_csv(stm_dir / "stm_results.csv")
    removed = set(map(int, stm_meta.get("docs_removed", [])))
    kept_indices = [i for i in range(len(original)) if i not in removed]
    aligned = original.iloc[kept_indices].reset_index(drop=True)
    stm_theta = theta
    if len(aligned) != len(stm_theta) or len(aligned) != len(stm_result):
        raise ValueError("Linhas mantidas pelo STM nao alinham com theta/resultados")

    all_tokens = original.text.fillna("").str.split().tolist()
    video_ids = original.post_id.astype(str).tolist()
    all_windows = context_windows(all_tokens, video_ids, radius=int(model_cfg["context_radius"]))
    if len(all_windows) != len(original):
        raise AssertionError("Uma janela por frase original era esperada")
    documents = [all_windows[i] for i in kept_indices]

    seeds_path = TOPIC / "configs" / cfg["seeds_file"]
    raw_seeds = load_seeds(seeds_path)
    seeds = normalize_seeds(
        raw_seeds,
        extra_stopwords=corpus_cfg.get("stopwords_emojis"),
        preserve_terms=model_cfg.get("preserve_seed_terms"),
    )
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=corpus_cfg["stm_no_below"],
                               no_above=corpus_cfg["stm_no_above"])
    effective, exclusions = apply_seed_exclusion_rule(
        seeds, dictionary, cfg["min_df_semente"]["youtube_sent"],
        cfg["min_sementes_por_aspecto"],
    )
    if not effective:
        raise RuntimeError("Nenhum aspecto semeado sobreviveu aos filtros")
    Y, aspect_names = build_seed_matrix(effective, dictionary)
    rows: list[int] = []
    columns: list[int] = []
    values: list[int] = []
    for row, doc in enumerate(documents):
        for token_id, count in dictionary.doc2bow(doc):
            rows.append(row)
            columns.append(token_id)
            values.append(count)
    counts = sparse.csr_matrix((values, (rows, columns)),
                               shape=(len(documents), len(dictionary)), dtype=np.float32)
    X = TfidfTransformer().fit_transform(counts).T.tocsr()
    del counts, rows, columns, values

    alpha_grid = model_cfg["alpha_grid"]
    lambda_grid = scaled_lambda_grid(X, Y, alpha_grid)
    diagnostics: list[dict] = []
    main_free = int(model_cfg["livres"])
    main_seed = int(cfg["seed_runs"][0])
    lam = escolher_lambda(
        X, Y, k=len(aspect_names) + main_free, seed=main_seed,
        grid=lambda_grid, max_iter=500, diagnostics=diagnostics,
        top_n=int(model_cfg["regra_lambda_top_n"]),
        frac_min=float(model_cfg["regra_lambda_frac"]),
    )
    lambda_rule_met = any(
        row["min_seed_recall"] >= float(model_cfg["regra_lambda_frac"])
        for row in diagnostics
    )

    output = make_run_output_dir(TOPIC / "data/output/youtube_sent/guided", "gnmf_context_v3")
    (output / "seeds_efetivas.json").write_text(
        json.dumps({"aspectos": effective, **exclusions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stm_dominant = stm_theta.argmax(axis=1)
    aspect_topics = list(map(int, model_cfg["aspect_topics"]))
    if not aspect_topics or max(aspect_topics) >= stm_theta.shape[1]:
        raise ValueError("Topicos STM de aspecto fora do intervalo da theta")
    categories = aligned.product_category.fillna("outras").astype(str).tolist()
    center_video_ids = aligned.post_id.astype(str).tolist()
    stm_cv = category_dependence(stm_theta, categories, center_video_ids)
    stm_cv = stm_cv.drop_duplicates("topico").set_index("topico")["cv"]

    runs: dict[str, dict] = {}
    free_values = [main_free, *map(int, model_cfg["livres_sens"])] if lambda_rule_met else [main_free]
    for n_free in free_values:
        for seed in map(int, cfg["seed_runs"]):
            subdir = output / f"livres_{n_free}_seed_{seed}"
            subdir.mkdir(parents=True, exist_ok=True)
            result = guided_nmf(X, Y, k=len(aspect_names) + n_free,
                                lam=lam, seed=seed, max_iter=500)
            A, S, B = result["A"], result["S"], result["B"]
            assignment = seed_topic_assignment(B)
            seed_recall = [
                len(set(np.argsort(-A[:, assignment[j]])[:int(model_cfg["regra_lambda_top_n"])])
                    & set(np.flatnonzero(Y[:, j]))) / int(np.count_nonzero(Y[:, j]))
                for j in range(len(aspect_names))
            ]
            dominant = S.argmax(axis=0)
            normalized = S / np.maximum(S.sum(axis=0, keepdims=True), 1e-12)
            is_seeded = np.isin(dominant, assignment)
            pd.DataFrame({"topico": dominant, "peso": normalized.max(axis=0),
                          "topico_semeado": is_seeded}).to_csv(subdir / "w_sentenca.csv", index=False)
            topic_rows = []
            for topic in range(A.shape[1]):
                aspect = next((aspect_names[j] for j, t in enumerate(assignment) if t == topic), None)
                for rank, word_id in enumerate(np.argsort(-A[:, topic])[:20], start=1):
                    word = dictionary[int(word_id)]
                    topic_rows.append({"topico": topic, "aspecto": aspect, "rank": rank,
                                       "palavra": word, "peso": float(A[word_id, topic]),
                                       "semente": bool(aspect and word in effective[aspect])})
            pd.DataFrame(topic_rows).to_csv(subdir / "topics.csv", index=False)
            overall = topic_concordance(stm_dominant, dominant)
            selected = np.isin(stm_dominant, aspect_topics)
            purity = purity_by_group(dominant[selected], stm_dominant[selected],
                                     allowed_topics=assignment)
            purity.to_csv(subdir / "concordancia_stm.csv", index=False)
            pd.crosstab(stm_dominant[selected], dominant[selected],
                        rownames=["stm"], colnames=["guided"]).to_csv(subdir / "contingencia_stm.csv")
            by_category = category_dependence(normalized.T, categories, center_video_ids)
            by_category.to_csv(subdir / "prevalencia_categoria.csv", index=False)
            guided_cv = by_category.drop_duplicates("topico").set_index("topico")["cv"]
            comparable = (purity[purity.topico_majoritario.isin(assignment)]
                          .sort_values("fracao_majoritaria", ascending=False)
                          .drop_duplicates("topico_majoritario"))
            cv_pairs = [(float(stm_cv.loc[int(row.grupo)]),
                         float(guided_cv.loc[int(row.topico_majoritario)]))
                        for row in comparable.itertuples()]
            from scipy.stats import spearmanr
            rho = float(spearmanr(*zip(*cv_pairs)).statistic) if len(cv_pairs) >= 2 else float("nan")
            pd.DataFrame([{"spearman_cv": rho, "n_aspectos_comuns": len(cv_pairs)}]).to_csv(
                subdir / "spearman_cv_stm.csv", index=False)
            run = {
                "dir": str(subdir), "nmi": overall["nmi"], "ari": overall["ari"],
                "aspect_nmi": topic_concordance(stm_dominant[selected], dominant[selected])["nmi"],
                "aspect_ari": topic_concordance(stm_dominant[selected], dominant[selected])["ari"],
                "pureza_ge_05": int((purity.fracao_majoritaria >= 0.5).sum()),
                "seeded_dominant_share": float(is_seeded.mean()),
                "min_seed_recall": float(min(seed_recall)),
                "loss_final": float(result["loss_history"][-1]),
            }
            runs[f"livres_{n_free}_seed_{seed}"] = run
            print(json.dumps({"run": f"livres_{n_free}_seed_{seed}", **run}, ensure_ascii=False), flush=True)
            del result, A, S, B, normalized

    metadata = {
        "protocol": model_cfg["protocol"], "stm_run": model_cfg["stm_run"],
        "context_radius": int(model_cfg["context_radius"]),
        "context_source": "full_stm_input_before_docs_removed; same-video neighbors",
        "n_original": len(original), "n_retained_centers": len(aligned),
        "vocab_size": len(dictionary), "seed_file_sha256": hashlib.sha256(seeds_path.read_bytes()).hexdigest(),
        "effective_seeds": effective, "exclusions": exclusions,
        "lambda": lam, "alpha_grid": alpha_grid, "lambda_grid": lambda_grid,
        "lambda_diagnostics": diagnostics, "lambda_rule_met": lambda_rule_met,
        "h6_gate_pass": all(
            runs[f"livres_{main_free}_seed_{seed}"]["pureza_ge_05"] >= 6
            for seed in map(int, cfg["seed_runs"])
        ),
        "runs": runs,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Output:", output, flush=True)
    print("H6 gate:", "PASS" if metadata["h6_gate_pass"] else "FAIL", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
