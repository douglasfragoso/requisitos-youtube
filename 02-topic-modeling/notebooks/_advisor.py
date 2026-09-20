"""LLM advisor for hyperparameter selection (calibration tool).

Non-destructive: proposes params.yaml patches + prose pre-analysis; never
writes params.yaml, never triggers sweeps. Providers: OpenAI (GPT) and Ollama
(local, via OpenAI-compat endpoint) — Anthropic is NOT used. See
docs/superpowers/specs/2026-07-06-llm-param-selection-design.md
"""
from __future__ import annotations

import fnmatch
import glob
import json
import os
import re

import yaml

# --- Prompts externalizados (configs/advisor_prompts.yaml) -------------------
_PROMPTS_CACHE: dict | None = None


def load_advisor_prompts(config_dir: str | None = None) -> dict:
    """Carrega os prompts de configs/advisor_prompts/ (um arquivo por prompt).

    Layout: advisor_prompts/shared.yaml + advisor_prompts/<metodo>/{system,
    varredura,final}.yaml (cada um com uma chave `prompt`). Devolve a mesma
    forma de sempre: {"shared": {...}, "methods": {m: {phase: str}}}.
    Cache em memória. Default config_dir: ../configs relativo a este arquivo.
    """
    global _PROMPTS_CACHE
    use_cache = config_dir is None          # cache só o carregamento default
    if use_cache and _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    if config_dir is None:
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")
    base = os.path.join(config_dir, "advisor_prompts")
    with open(os.path.join(base, "shared.yaml"), encoding="utf-8") as f:
        shared = yaml.safe_load(f)
    methods: dict = {}
    for m in sorted(os.listdir(base)):
        mdir = os.path.join(base, m)
        if not os.path.isdir(mdir):
            continue
        methods[m] = {}
        # Fases opcionais: um metodo pode ter so um subconjunto (ex. 'comparacao'
        # tem system+site, sem varredura/final). Ausencia nao e erro.
        # `system_site`: persona da fase site. O `system` foi escrito para a
        # calibração de hiperparâmetros ("propõe a próxima config") e contradizia
        # o site.yaml ("sua tarefa NÃO é propor hiperparâmetros").
        for phase in ("system", "system_site", "varredura", "final", "site"):
            fp = os.path.join(mdir, f"{phase}.yaml")
            if not os.path.exists(fp):
                continue
            with open(fp, encoding="utf-8") as f:
                methods[m][phase] = yaml.safe_load(f)["prompt"]
    data = {"shared": shared, "methods": methods}
    if use_cache:
        _PROMPTS_CACHE = data
    return data


# --- Allowlist: exact params.yaml paths the advisor may propose --------------
ALLOWED_BERTOPIC_PATHS = {
    "bertopic.umap.n_neighbors",
    "bertopic.hdbscan.min_cluster_size",
    "bertopic.hdbscan.min_samples",
    "bertopic.hdbscan.cluster_selection_method",
    "bertopic.reduce_topics_nr",
    "bertopic.macro_k",
    "bertopic.reduce_outliers.strategy",
    "bertopic.reduce_outliers.threshold",
}
ALLOWED_OTHER_PATHS = {
    "lda.k_range", "lda.no_below", "lda.no_above",
    "nmf.k_range", "nmf.no_below", "nmf.no_above", "nmf.passes",
    "evaluation.lda_alpha_grid", "evaluation.lda_eta_grid",
    "evaluation.nmf_kappa_grid", "evaluation.nmf_min_prob_grid",
    "evaluation.run_param_sweep",
    "evaluation.run_outlier_sweep",
    "evaluation.run_threshold_sweep",
    "evaluation.param_sweep_n_neighbors",
    "evaluation.param_sweep_min_cluster_size",
    "evaluation.param_sweep_min_samples",
    "evaluation.param_sweep_cluster_selection_methods",
    "evaluation.param_sweep_reduce_topics_nr",
    "evaluation.outlier_sweep_strategies",
    "evaluation.threshold_sweep_grids",
    "evaluation.stm_sigma_prior_grid", "evaluation.stm_gamma_prior_grid",
}
# STM: só hiperparâmetros científicos (grid de K + qualidade de treino). NUNCA
# rscript_path/timeout_sec — são configuração de máquina/operacional, não
# entram aqui de propósito (ver Global Constraints).
ALLOWED_STM_PATHS = {
    "stm.k_range", "stm.no_below", "stm.no_above", "stm.max_em_its",
}
# Pins por corpus que desligam os grids do STM nos notebooks (params.yaml >
# corpora.<id>.stm_best_k/stm_sigma_prior/stm_gamma_prior) — não seguem o
# padrão bertopic_overrides.<suffix>; checados à parte em _is_allowed_key.
ALLOWED_STM_CORPUS_PIN_KEYS = {"stm_best_k", "stm_sigma_prior", "stm_gamma_prior"}
# Declared in params.yaml but inert (no _helpers function consumes it).
FORBIDDEN_PATHS = {"evaluation.param_sweep_macro_k"}

PARAMS_ALLOWLIST = ALLOWED_BERTOPIC_PATHS | ALLOWED_OTHER_PATHS | ALLOWED_STM_PATHS


def _is_allowed_key(key: str) -> bool:
    if key in FORBIDDEN_PATHS:
        return False
    if key in PARAMS_ALLOWLIST:
        return True
    parts = key.split(".")
    # corpora.<corpus>.bertopic_overrides.<suffix> mirrors bertopic.<suffix>
    if len(parts) >= 4 and parts[0] == "corpora" and parts[2] == "bertopic_overrides":
        suffix = ".".join(parts[3:])
        return f"bertopic.{suffix}" in ALLOWED_BERTOPIC_PATHS
    # corpora.<corpus>.stm_best_k / stm_sigma_prior / stm_gamma_prior (pins)
    if len(parts) == 3 and parts[0] == "corpora" and parts[2] in ALLOWED_STM_CORPUS_PIN_KEYS:
        return True
    return False


# --- Value type/range validation --------------------------------------------
# Um valor pode ter o NOME certo (allowlist) mas o TIPO errado — vimos
# `evaluation.param_sweep_min_samples: true` (tem de ser LISTA). O spec abaixo
# checa tipo/faixa por chave; quem passa no nome mas falha no valor vai p/
# invalid_keys (não é aplicado). Chaves sem spec são só validadas por nome.
def _is_bool(v) -> bool:
    return isinstance(v, bool)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int_pos(v) -> bool:
    return _is_int(v) and v >= 1


_STRATEGIES = ("off", "c-tf-idf", "embeddings", "probabilities", "distributions")
# Specs por chave NORMALIZADA (per-corpus overrides/pins caem no mesmo spec do
# caminho global via _normalize_spec_key).
_VALUE_SPEC = {
    "bertopic.umap.n_neighbors": _int_pos,
    "bertopic.hdbscan.min_cluster_size": _int_pos,
    "bertopic.hdbscan.min_samples": lambda v: v is None or _int_pos(v),
    "bertopic.hdbscan.cluster_selection_method": lambda v: v in ("eom", "leaf"),
    "bertopic.reduce_topics_nr": _int_pos,
    "bertopic.macro_k": _int_pos,
    "bertopic.reduce_outliers.strategy": lambda v: v in _STRATEGIES,
    "bertopic.reduce_outliers.threshold": lambda v: _is_num(v) and 0 <= v <= 1,
    "lda.no_below": _int_pos, "nmf.no_below": _int_pos, "stm.no_below": _int_pos,
    "nmf.passes": _int_pos, "stm.max_em_its": _int_pos,
    "evaluation.run_param_sweep": _is_bool,
    "evaluation.run_outlier_sweep": _is_bool,
    "evaluation.run_threshold_sweep": _is_bool,
    # STM pins per-corpus (normalizados p/ o nome-base)
    "stm_best_k": _int_pos,
    "stm_sigma_prior": lambda v: _is_num(v) and 0 <= v <= 1,
    "stm_gamma_prior": lambda v: v in ("Pooled", "L1"),
}


def _normalize_spec_key(key: str) -> str:
    """Mapeia overrides/pins per-corpus para o caminho global equivalente."""
    parts = key.split(".")
    if len(parts) >= 4 and parts[0] == "corpora" and parts[2] == "bertopic_overrides":
        return "bertopic." + ".".join(parts[3:])
    if len(parts) == 3 and parts[0] == "corpora" and parts[2] in ALLOWED_STM_CORPUS_PIN_KEYS:
        return parts[2]
    return key


def _value_ok(key: str, value) -> bool:
    """True se o valor bate com o tipo/faixa esperado da chave (ou sem spec)."""
    nk = _normalize_spec_key(key)
    spec = _VALUE_SPEC.get(nk)
    if spec is not None:
        return bool(spec(value))
    # Regras por família de chave (listas): k_range, grids, param_sweep_*,
    # *_sweep_strategies, threshold_sweep_grids — todas têm de ser listas.
    if nk.endswith(".k_range"):
        return isinstance(value, list)
    if nk.startswith("evaluation."):
        if (nk.endswith("_grid") or nk.endswith("_grids")
                or nk.startswith("evaluation.param_sweep_")
                or nk.endswith("_sweep_strategies")):
            return isinstance(value, list)
    return True  # sem spec → só validação de nome (como antes)


def validate_patch(patch: dict) -> tuple[dict, list[str]]:
    """Split a proposed patch into (valid_keys, invalid_keys).

    Invalid = not in the allowlist, explicitly forbidden, OR name-allowed but
    with a value whose type/range fails its spec. Invalid keys are never
    applied; they are surfaced to the human.
    """
    valid: dict = {}
    invalid: list[str] = []
    for k, v in (patch or {}).items():
        if _is_allowed_key(k) and _value_ok(k, v):
            valid[k] = v
        else:
            invalid.append(k)
    return valid, invalid


# --- Task 2: parser da resposta do LLM ---------------------------------------
class ProposalFormatError(Exception):
    """Raised when the LLM response lacks a valid `yaml proposal` block."""


_PROPOSAL_RE = re.compile(r"```yaml proposal\s*\n(.*?)```", re.DOTALL)


def parse_proposal(response: str) -> tuple[dict, list[str], str]:
    m = _PROPOSAL_RE.search(response or "")
    if not m:
        raise ProposalFormatError("no 'yaml proposal' block found in response")
    try:
        block = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        raise ProposalFormatError(f"proposal block is not valid YAML: {exc}") from exc
    if not isinstance(block, dict) or "params_patch" not in block:
        raise ProposalFormatError("proposal block missing 'params_patch'")

    valid, invalid = validate_patch(block.get("params_patch") or {})
    block["params_patch"] = valid
    # Prosa = tudo FORA do bloco yaml (antes + depois). O modelo às vezes escreve
    # a ## Pré-análise ANTES do bloco; pegar só o depois perdia a prosa.
    report = (response[:m.start()] + "\n" + response[m.end():]).strip()
    return block, invalid, report


# --- Task 3: ledger e detecção de rodada -------------------------------------
def read_ledger(ledger_path: str) -> dict | None:
    if not os.path.exists(ledger_path):
        return None
    with open(ledger_path, encoding="utf-8") as f:
        return json.load(f)


def resolve_round(ledger_path: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    led = read_ledger(ledger_path)
    if not led or not led.get("rounds"):
        return 1
    return max(int(r["round"]) for r in led["rounds"]) + 1


def append_ledger(ledger_path: str, corpus: str, model: str, entry: dict) -> None:
    led = read_ledger(ledger_path) or {"corpus": corpus, "model": model, "rounds": []}
    led["rounds"].append(entry)
    os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(led, f, ensure_ascii=False, indent=2)


# --- Evolução: memória de chat (advisor_chat.json) ---------------------------
def read_chat(chat_path: str) -> list:
    """Lista de mensagens [{role, content}] persistida (ou [] se ausente)."""
    if not os.path.exists(chat_path):
        return []
    with open(chat_path, encoding="utf-8") as f:
        return json.load(f).get("messages", [])


def save_chat(chat_path: str, corpus: str, model: str, messages: list) -> None:
    os.makedirs(os.path.dirname(chat_path) or ".", exist_ok=True)
    with open(chat_path, "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus, "model": model, "messages": messages},
                  f, ensure_ascii=False, indent=2)


# --- Task 4: coleta de evidências --------------------------------------------
_EVIDENCE_GLOBS = (
    "*metrics.csv",            # tweets/bertopic + LDA + NMF + STM; AUSENTE em folha/bertopic (ok)
    "*topics_for_eval.csv",
    "*robustness*.csv",
    "sweep_*.csv",            # sweeps estruturais/outlier — geralmente num run SEM results.csv
    "*macro_k_sweep.csv",     # não casa com sweep_*.csv (prefixo 'bertopic_'); listar à parte
    "*exclusividade_ranking.csv",
    "*_grid.csv",             # caches de grid LDA/NMF/STM na pasta-BASE (lda_alpha_eta_grid,
                              # nmf_kappa_minprob_grid, stm_sigma_gamma_grid); sweep_bertopic_grid
                              # já coberto acima
    "*grid_k_diagnostics.csv",  # STM: heldout likelihood/semantic coherence por K
                                # (stm_grid_k_diagnostics.csv) — não casa com *_grid.csv
                                # (sufixo é _diagnostics.csv, não _grid.csv); listado à parte
    # --- fase site: dados por trás de visualizações específicas ---
    "*topics_frex.csv",         # STM: rótulos FREX vs. maior probabilidade
    "*macro_temas.csv",         # BERTopic: agrupamento Ward dos tópicos
    "*categoria_topico.csv",    # BERTopic: cross-tab nativo tópico × categoria
    "*_beta.csv",               # NMF/STM: matriz tópico×vocabulário (heatmap phi/beta);
                                # LARGA (K linhas × milhares de colunas) — truncada por
                                # COLUNA em _truncate_wide_csv, não por linha (ver Correção 1)
)
# Nomes de evidência tratados como MATRIZES LARGAS (linhas=tópicos, colunas=
# vocabulário): truncar por linha (_truncate_csv) não ajuda em nada aqui, o
# problema é a LARGURA. Usa fnmatch contra o próprio padrão do glob.
_WIDE_EVIDENCE_GLOBS = ("*_beta.csv",)


def _run_dirs_newest_first(base_output: str, corpus: str) -> list[str]:
    """Subdirs carimbados `<corpus>_<stamp>` do mais novo p/ o mais antigo.

    Ordena por nome (= timestamp), mesma convenção do resolve_latest_dir
    (`max(key=d.name)`). Robusto a mtime alterado por cópia.
    """
    dirs = [d for d in glob.glob(os.path.join(base_output, f"{corpus}_*")) if os.path.isdir(d)]
    return sorted(dirs, key=os.path.basename, reverse=True)


def _truncate_csv(path: str, max_rows: int) -> str:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    head = lines[: max_rows + 1]  # header + max_rows
    return "".join(head)


def _truncate_wide_csv(path: str, max_cols: int = 30) -> str:
    """Trunca uma matriz LARGA (K tópicos × milhares de palavras, ex.:
    nmf_beta.csv/stm_beta.csv) por COLUNA, não por linha.

    `_truncate_csv` (linhas) não ajuda aqui: um beta tem só ~20-25 linhas
    (1 por tópico) mas MILHARES de colunas (vocabulário inteiro) — é a
    LARGURA que estoura o snippet (folha/nmf: ~4.5MB; folha/stm: ~9.3MB
    para 25 tópicos × 17k+ colunas). Mantém as `max_cols` colunas de maior
    peso TOTAL (soma sobre os tópicos) — as palavras mais influentes do
    modelo inteiro, aproximação razoável do que o heatmap phi destaca.

    `nmf_beta.csv` já vem com coluna `topic_id`; `stm_beta.csv` não tem
    coluna de índice (1 linha por tópico, na ordem 0..K-1) — sintetiza
    `topic_id` a partir da posição da linha nesse caso.
    """
    import pandas as pd
    df = pd.read_csv(path)
    if "topic_id" in df.columns:
        id_col = df["topic_id"]
        values = df.drop(columns=["topic_id"])
    else:
        id_col = pd.Series(range(len(df)), name="topic_id")
        values = df
    top_cols = values.sum(axis=0).sort_values(ascending=False).head(max_cols).index
    out = pd.concat([id_col, values[top_cols]], axis=1)
    return out.to_csv(index=False)


def _derive_topic_ranking_from_results(results_path: str) -> str:
    """Backfill: n_docs por tópico a partir do *_results.csv (sem carregar o texto).

    Item 2 (híbrido): enquanto o pipeline do LDA/NMF/STM não emitir
    *_exclusividade_ranking.csv, o advisor deriva n_docs por tópico dominante
    aqui — agregação leve (só topic_id/topic_name/granularity). Devolve CSV
    (topic_id,topic_name,n_docs) ordenado por n_docs desc, ou "" se falhar.
    """
    import pandas as pd
    try:
        cols = pd.read_csv(results_path, nrows=0).columns
        use = [c for c in ("topic_id", "topic_name", "granularity") if c in cols]
        if "topic_id" not in use:
            return ""
        df = pd.read_csv(results_path, usecols=use)
    except Exception:
        return ""
    # Invariante: LDA/NMF/STM sempre exportam a coluna `granularity` (unit = 1
    # linha por doc; macro = agregados). Filtra p/ unit e evita dupla contagem.
    # Se um export futuro trouxer linhas macro SEM a coluna, o n_docs infla — por
    # isso o filtro é condicional à existência da coluna (hoje sempre presente).
    if "granularity" in df.columns:
        df = df[df["granularity"] == "unit"]
    keys = [c for c in ("topic_id", "topic_name") if c in df.columns]
    g = (df.groupby(keys).size().reset_index(name="n_docs")
         .sort_values("n_docs", ascending=False))
    return g.to_csv(index=False)


def _find_corpus_csv(corpus: str) -> str | None:
    """corpus_limpo.csv do run mais novo do 01-preprocessing (latest-wins).

    Resolve a partir da localização DESTE arquivo (não do cwd): o advisor roda
    tanto de notebooks/ quanto da raiz. Cai no layout plano legado
    (03-topic-modeling/data/input/<corpus>/) se o 01 não tiver output.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = os.path.join(root, "01-preprocessing", "data", "output", corpus)
    candidatos = sorted(
        (d for d in glob.glob(os.path.join(base, f"{corpus}_*")) if os.path.isdir(d)),
        key=os.path.basename, reverse=True)
    for d in candidatos:
        p = os.path.join(d, "corpus_limpo.csv")
        if os.path.exists(p):
            return p
    legado = os.path.join(root, "03-topic-modeling", "data", "input", corpus, "corpus_limpo.csv")
    return legado if os.path.exists(legado) else None


def _load_results_with_corpus(results_path: str, corpus_csv: str):
    """Junta *_results.csv (topic_id por post_id) com data/category do corpus."""
    import pandas as pd
    cols_res = pd.read_csv(results_path, nrows=0).columns
    use = [c for c in ("post_id", "topic_id", "topic_name", "granularity") if c in cols_res]
    if "post_id" not in use or "topic_id" not in use:
        return None
    res = pd.read_csv(results_path, usecols=use)
    if "granularity" in res.columns:
        res = res[res["granularity"] == "unit"]
    # Outliers do BERTopic fora. NÃO basta filtrar -1: os dois corpora divergem —
    # folha mantém id -1, mas tweets_bre2022 renumera o bucket residual para o
    # último id (24) com topic_name "Outlier". Sem o filtro por nome, 1452 docs de
    # outlier entravam no cross-tab e na timeline como se fossem um tópico real, e
    # o assessor chegou a atribuir "forte presença" a esse pseudo-tópico.
    res = res[res["topic_id"] != -1]
    if "topic_name" in res.columns:
        res = res[~res["topic_name"].astype(str).str.fullmatch(r"\s*outlier\s*", case=False, na=False)]
    cols_cor = pd.read_csv(corpus_csv, nrows=0).columns
    use_cor = [c for c in ("post_id", "data", "date", "category") if c in cols_cor]
    if "post_id" not in use_cor:
        return None
    cor = pd.read_csv(corpus_csv, usecols=use_cor)
    df = res.merge(cor, on="post_id", how="inner")
    return df if not df.empty else None


def _derive_topic_category_from_results(results_path: str, corpus_csv: str) -> str:
    """Cross-tab tópico × categoria em % da linha (o que o heatmap do site mostra).

    Formato largo (categorias nas linhas, tópicos nas colunas) para caber na
    evidência sem truncamento. Devolve "" se não der para derivar.
    """
    try:
        df = _load_results_with_corpus(results_path, corpus_csv)
        if df is None or "category" not in df.columns:
            return ""
        pivot = df.groupby(["category", "topic_id"]).size().unstack(fill_value=0)
        if pivot.empty:
            return ""
        n_docs = pivot.sum(axis=1)
        pct = pivot.div(n_docs, axis=0).mul(100).round(1)
        pct.insert(0, "n_docs", n_docs)
        pct.columns = ["n_docs"] + [f"T{c}" for c in pct.columns[1:]]
        return pct.reset_index().to_csv(index=False)
    except Exception:
        return ""


def _derive_topic_timeline_from_results(results_path: str, corpus_csv: str) -> str:
    """Série temporal de documentos por tópico (o que os gráficos de evolução mostram).

    Granularidade SEMPRE mensal — é a das figuras publicadas. Antes agregava por
    ano acima de 24 períodos, e a folha (81 meses) caía nisso: o LLM recebia 9
    linhas anuais, escrevia sobre "eventos específicos" que só existem no recorte
    mensal, e o leitor via um gráfico mensal ao lado. 81 linhas de CSV é barato
    perto de publicar uma leitura que não corresponde à figura.
    Devolve "" se não der para derivar.
    """
    import pandas as pd
    try:
        df = _load_results_with_corpus(results_path, corpus_csv)
        if df is None:
            return ""
        col_data = next((c for c in ("data", "date") if c in df.columns), None)
        if col_data is None:
            return ""
        dt = pd.to_datetime(df[col_data], errors="coerce")
        df = df.assign(_dt=dt).dropna(subset=["_dt"])
        if df.empty:
            return ""
        rotulo = "mes"
        periodo = df["_dt"].dt.to_period("M")
        tab = (df.assign(**{rotulo: periodo.astype(str)})
                 .groupby([rotulo, "topic_id"]).size().unstack(fill_value=0)
                 .sort_index())
        tab.columns = [f"T{c}" for c in tab.columns]
        return tab.reset_index().to_csv(index=False)
    except Exception:
        return ""


def _bh_ajustar(p):
    """p ajustado por Benjamini-Hochberg (FDR). Sem dependência nova."""
    import numpy as np
    p = np.asarray(p, dtype=float)
    n = p.size
    ordem = np.argsort(p)
    ajust = np.empty(n, dtype=float)
    # BH: p_aj[(i)] = min_{j>=i} ( n/j * p[(j)] ), varrendo de trás para frente
    anterior = 1.0
    for pos in range(n - 1, -1, -1):
        idx = ordem[pos]
        anterior = min(anterior, p[idx] * n / (pos + 1))
        ajust[idx] = anterior
    return np.clip(ajust, 0.0, 1.0)


def _derive_stm_prevalence_summary(effects_path: str, max_linhas: int = 40) -> str:
    """Resumo do estimateEffect: efeitos significativos por FDR (Benjamini-Hochberg).

    O `stm_prevalence_effects.csv` tem ~500 linhas (todos os termos × tópicos) —
    truncado nas 40 primeiras cobriria 2 tópicos. Aqui descarta o intercepto,
    filtra e ordena por |estimate|: é o que o forest plot destaca.

    O corte usa p AJUSTADO, não o bruto: são ~225 testes (25 tópicos × 9
    categorias na folha) e, a α=0,05 sem correção, ~11 falsos positivos são
    esperados só por acaso — a lista rotulada "significativos" inflava a
    confiança de quem lê. Devolve "" se falhar.
    """
    import pandas as pd
    try:
        df = pd.read_csv(effects_path)
        need = {"topic_id", "term", "estimate", "p_value"}
        if not need.issubset(df.columns):
            return ""
        df = df[~df["term"].astype(str).str.contains(r"\(Intercept\)", regex=True)]
        df = df.assign(p_value=pd.to_numeric(df["p_value"], errors="coerce")).dropna(
            subset=["p_value"])
        if df.empty:
            return ""
        df = df.assign(p_ajustado_bh=_bh_ajustar(df["p_value"].to_numpy()).round(5))
        df = df[df["p_ajustado_bh"] < 0.05]
        if df.empty:
            return ""
        df = df.reindex(df["estimate"].abs().sort_values(ascending=False).index)
        return df.head(max_linhas).to_csv(index=False)
    except Exception:
        return ""


def collect_evidence(corpus: str, model: str, base_output: str, *, max_rows: int = 40,
                     corpus_info: str = "") -> dict:
    run_dirs = _run_dirs_newest_first(base_output, corpus)
    # Proveniência: run "primário" = mais novo com results.csv (modelo de produção);
    # se nenhum tiver, o mais novo no geral; se não houver run, o próprio base_output.
    primary = next(
        (d for d in run_dirs if glob.glob(os.path.join(d, "*results.csv"))),
        run_dirs[0] if run_dirs else base_output,
    )
    # Para CADA arquivo de evidência DISTINTO, pega a versão do run carimbado MAIS
    # NOVO que o contém (run_dirs vem newest-first + 1º vence). base_output entra
    # por ÚLTIMO: é onde vivem os caches de grid do LDA/NMF/STM.
    snippets: dict[str, str] = {}
    for pattern in _EVIDENCE_GLOBS:
        for d in run_dirs + [base_output]:  # newest-first; base por último
            for path in sorted(glob.glob(os.path.join(d, pattern))):
                name = os.path.basename(path)
                if name.endswith("_raw.csv") or name in snippets:
                    continue  # _raw (per-seed) = ruído; nome já visto = versão mais nova
                if any(fnmatch.fnmatch(name, p) for p in _WIDE_EVIDENCE_GLOBS):
                    snippets[name] = _truncate_wide_csv(path)  # trunca por COLUNA
                else:
                    snippets[name] = _truncate_csv(path, max_rows)
    # Backfill item 2 (híbrido): p/ LDA/NMF/STM, se o pipeline ainda NÃO emite
    # *_exclusividade_ranking.csv, deriva n_docs por tópico do *_results.csv.
    # Quando o pipeline passar a emitir o ranking canônico, este bloco não roda
    # (a condição `not any(... exclusividade_ranking ...)` falha).
    if (model.lower() in ("lda", "nmf", "stm")
            and not any(n.endswith("exclusividade_ranking.csv") for n in snippets)):
        results = next(
            (p for d in run_dirs + [base_output]
             for p in sorted(glob.glob(os.path.join(d, "*results.csv")))),
            None,
        )
        if results:
            derived = _derive_topic_ranking_from_results(results)
            if derived:
                snippets["derived_topic_ranking.csv"] = derived
    # Fase site: as visualizações de cross-tab (tópico × editoria/mês) e de
    # evolução temporal só existem como PNG — sem CSV, a auditoria devolvia
    # "(dados não disponíveis nesta rodada)". Deriva as duas tabelas do
    # *_results.csv + data/category do corpus (vale p/ os 4 modelos).
    _results = next(
        (p for d in run_dirs + [base_output]
         for p in sorted(glob.glob(os.path.join(d, "*results.csv")))),
        None,
    )
    # STM: efeitos de prevalência resumidos (o arquivo cru tem ~500 linhas)
    _efeitos = next(
        (p for d in run_dirs + [base_output]
         for p in sorted(glob.glob(os.path.join(d, "*prevalence_effects.csv")))),
        None,
    )
    if _efeitos:
        resumo = _derive_stm_prevalence_summary(_efeitos)
        if resumo:
            snippets["derived_stm_prevalence_significativos.csv"] = resumo
    _corpus_csv = _find_corpus_csv(corpus) if _results else None
    if _results and _corpus_csv:
        for nome, fn in (("derived_topic_category.csv", _derive_topic_category_from_results),
                         ("derived_topic_timeline.csv", _derive_topic_timeline_from_results)):
            if nome not in snippets:
                derivado = fn(_results, _corpus_csv)
                if derivado:
                    snippets[nome] = derivado
    led = read_ledger(os.path.join(base_output, "llm_advisor_state.json"))
    return {
        "corpus": corpus,
        "model": model,
        "corpus_info": corpus_info,
        "run_dir": primary,
        "csv_snippets": snippets,
        "ledger_rounds": len(led["rounds"]) if led else 0,
    }


# --- Task 5 / Evolução: montagem das mensagens (prompts vêm do YAML) ---------
def _allowlist_for_model(model_type: str, corpus: str | None = None) -> list[str]:
    """Caminhos exatos de params.yaml que o assessor pode propor para o modelo.

    Injetado no prompt para o LLM usar os nomes CERTOS (senão ele chuta variações
    como `evaluation.lda.alpha` em vez de `evaluation.lda_alpha_grid`, e o
    allowlist rejeita — patch válido vazio).
    """
    m = (model_type or "").lower()
    if m == "lda":
        return sorted({"lda.k_range", "lda.no_below", "lda.no_above",
                       "evaluation.lda_alpha_grid", "evaluation.lda_eta_grid"})
    if m == "nmf":
        return sorted({"nmf.k_range", "nmf.no_below", "nmf.no_above", "nmf.passes",
                       "evaluation.nmf_kappa_grid", "evaluation.nmf_min_prob_grid"})
    if m == "stm":
        base = {"stm.k_range", "stm.no_below", "stm.no_above", "stm.max_em_its",
                "evaluation.stm_sigma_prior_grid", "evaluation.stm_gamma_prior_grid"}
        if corpus:
            base |= {f"corpora.{corpus}.{k}" for k in ALLOWED_STM_CORPUS_PIN_KEYS}
        return sorted(base)
    if m == "bertopic":
        base = set(ALLOWED_BERTOPIC_PATHS) | {
            "evaluation.run_param_sweep", "evaluation.run_outlier_sweep",
            "evaluation.run_threshold_sweep",
            "evaluation.param_sweep_n_neighbors", "evaluation.param_sweep_min_cluster_size",
            "evaluation.param_sweep_min_samples", "evaluation.param_sweep_cluster_selection_methods",
            "evaluation.param_sweep_reduce_topics_nr", "evaluation.outlier_sweep_strategies",
            "evaluation.threshold_sweep_grids",
        }
        if corpus:
            base |= {f"corpora.{corpus}.bertopic_overrides.{p.split('.', 1)[1]}"
                     for p in ALLOWED_BERTOPIC_PATHS}
        return sorted(base)
    return sorted(PARAMS_ALLOWLIST)


def select_phase(round_n: int, rounds: int = 3, override: str | None = None) -> str:
    """Fase da rodada: 'final' quando round_n >= rounds; senão 'varredura'."""
    if override in ("varredura", "final"):
        return override
    return "final" if round_n >= rounds else "varredura"


def _fill(template: str, tokens: dict) -> str:
    """Substitui {{TOKEN}} por str.replace (NÃO str.format — conteúdo tem {}/CSV).

    Ordem: os FRAGMENTOS grandes (OUTPUT_CONTRACT/EVAL_METRICS) são injetados
    PRIMEIRO, para que os tokens escalares ({{ROUND}}, {{CORPUS}}, ...) resolvam
    tokens aninhados DENTRO deles (ex.: `round: {{ROUND}}` escrito no
    output_contract). {{EVIDENCE}} fica por ÚLTIMO — conteúdo de CSV nunca é
    re-varrido.
    """
    out = template
    for k in ("OUTPUT_CONTRACT", "EVAL_METRICS", "VIZ_AUDIT_CONTRACT",
              "CORPUS", "CORPUS_INFO", "ROUND", "ALLOWLIST", "VIZ_LIST",
              "EVIDENCE"):
        if k in tokens:
            out = out.replace("{{" + k + "}}", str(tokens[k]))
    return out


def build_advisor_messages(evidence, round_n, model_type, *, phase=None,
                           history=None, prompts=None, rounds=3):
    """Lista de mensagens a ENVIAR: [{system}, {user}] na 1ª rodada; senão
    history + [{user}]. O turno user vem do template da fase (varredura|final)."""
    prompts = prompts or load_advisor_prompts()
    m = (model_type or "").lower()
    if m not in prompts["methods"]:
        raise AdvisorConfigError(
            f"modelo desconhecido {m!r} — esperado um de {sorted(prompts['methods'])}")
    method = prompts["methods"][m]
    shared = prompts["shared"]
    phase = phase or select_phase(round_n, rounds=rounds)

    snippets = "\n\n".join(
        f"### {name}\n```csv\n{text}```" for name, text in evidence["csv_snippets"].items()
    ) or "(sem CSVs — provável rodada 1 de diagnóstico)"
    allowed = _allowlist_for_model(model_type, evidence.get("corpus"))
    allow_block = "\n".join(f"  - {p}" for p in allowed)

    tokens = {"CORPUS": evidence.get("corpus", "?"),
              "CORPUS_INFO": evidence.get("corpus_info", ""), "ROUND": round_n,
              "ALLOWLIST": allow_block, "EVIDENCE": snippets,
              "OUTPUT_CONTRACT": shared["output_contract"]}
    user_msg = {"role": "user", "content": _fill(method[phase], tokens)}
    system_msg = {"role": "system",
                  "content": _fill(method["system"], {"EVAL_METRICS": shared["eval_metrics"]})}

    if history:
        # normalmente o histórico já começa com o system; se não (chat.json antigo/
        # editado à mão/truncado), prepende o system para não mandar sem contexto.
        if history[0].get("role") == "system":
            return list(history) + [user_msg]
        return [system_msg] + list(history) + [user_msg]
    return [system_msg, user_msg]


# --- Task 6: chamada à API (OpenAI primary + Ollama fallback) ----------------
# --- Fase site: auditoria de visualizações (dados por trás de cada viz) ------
# Slugs estáveis: o site inclui <run_dir>/advisor_viz/<slug>.md literalmente.
VIZ_SLUGS: dict[str, list[tuple[str, str]]] = {
    "lda": [
        ("pyldavis", "pyLDAvis — mapa intertópico e termos por relevância"),
        ("heatmap_phi", "Heatmap φ (palavra × tópico)"),
        ("similaridade_topicos", "Similaridade cosseno entre tópicos"),
        ("docs_por_topico", "Distribuição de documentos por tópico"),
        ("topico_categoria", "Cross-tab tópico × categoria (editoria/mês)"),
        ("tsne_theta", "t-SNE do espaço θ"),
        ("grid_k", "Curva C_v × K (grid de K)"),
        ("grid_priors", "Grid α × η (C_v e perplexidade)"),
    ],
    "nmf": [
        ("keywords_barchart", "Top keywords por tópico (pesos H)"),
        ("heatmap_phi", "Heatmap H (palavra × tópico)"),
        ("similaridade_topicos", "Similaridade cosseno entre tópicos"),
        ("docs_por_topico", "Distribuição de documentos por tópico"),
        ("topico_categoria", "Cross-tab tópico × categoria (editoria/mês)"),
        ("tsne_theta", "t-SNE do espaço θ"),
        ("grid_k", "Curva C_v × K (grid de K)"),
        ("grid_hparams", "Grid kappa × minimum_probability"),
    ],
    "stm": [
        ("prevalencia_covariaveis", "Efeitos de prevalência (estimateEffect)"),
        ("prevalencia_temporal", "Prevalência esperada ao longo do tempo"),
        ("topics_over_time", "Evolução temporal dos tópicos"),
        ("frex", "Rótulos FREX vs. maior probabilidade"),
        ("heatmap_phi", "Heatmap φ (palavra × tópico)"),
        ("similaridade_topicos", "Similaridade entre tópicos"),
        ("topico_categoria", "Cross-tab tópico × categoria"),
        ("grid_k_diagnosticos", "Diagnósticos do grid de K (C_v, heldout, resíduos, semcoh × exclus)"),
    ],
    "bertopic": [
        ("topic_map", "Mapa intertópico (distância entre tópicos)"),
        ("hierarquia", "Dendrograma hierárquico de tópicos"),
        ("keywords", "Keywords c-TF-IDF por tópico (barchart/wordclouds)"),
        ("sankey_reduce", "Sankey do reduce_topics"),
        ("documents_umap", "Documentos no espaço UMAP"),
        ("similaridade_topicos", "Similaridade cosseno entre tópicos"),
        ("docs_por_topico", "Distribuição de documentos por tópico e outliers"),
        ("topico_categoria", "Cross-tab tópico × categoria (quando houver ground truth)"),
        ("evolucao_temporal", "Evolução temporal dos tópicos"),
        ("macro_temas", "Macro-temas (Ward sobre embeddings de tópico)"),
        ("robustez_topn", "Robustez das métricas por top-N"),
    ],
    "comparacao": [
        ("metricas_comparadas", "C_v, Topic Diversity, Exclusividade e Cobertura por modelo × corpus"),
        ("trade_offs", "Trade-offs entre modelos (cobertura × exclusividade; BoW × embeddings)"),
        ("leitura_por_corpus", "Qual modelo lê melhor cada corpus, e por quê"),
    ],
}


PLACEHOLDER_SEM_DADOS = "(dados não disponíveis nesta rodada)"

# Qual evidência sustenta qual visualização. Slug AUSENTE deste mapa (ou com
# lista vazia) não tem CSV por trás e NUNCA é pedido ao LLM — recebe o
# placeholder em código.
#
# Existe porque despejar todos os CSVs num bloco único e listar só `slug: título`
# obrigava o modelo a adivinhar o pareamento, e ele errou em produção: em
# folha/bertopic/robustez_topn.md leu a coluna `top_n` (nº de palavras por
# tópico, K fixo em 24) como "número de tópicos" e concluiu que "fragmentação
# excessiva degrada a qualidade" — conclusão espúria, publicada, contradizendo o
# partial logo acima dela. O mesmo slug saiu correto no corpus tweets: era
# adivinhação, não erro sistemático.
#
# Valores: (padrões de nome de CSV, glossário de colunas ambíguas).
_SEM_CSV: tuple = ((), "")
VIZ_EVIDENCE: dict[str, dict[str, tuple]] = {
    "lda": {
        "pyldavis": (("*topics_for_eval.csv", "*exclusividade_ranking.csv"), ""),
        "heatmap_phi": (("*topics_for_eval.csv",),
                        "só as top keywords por tópico; a matriz φ com as "
                        "probabilidades NÃO está disponível — não afirme em "
                        "quais tópicos um termo aparece além do que as keywords mostram"),
        "similaridade_topicos": _SEM_CSV,
        "docs_por_topico": (("*exclusividade_ranking.csv", "*metrics.csv",
                             "derived_topic_ranking.csv"),
                            "n_docs = documentos cujo tópico dominante é esse"),
        "topico_categoria": (("derived_topic_category.csv",),
                             "valores em % da LINHA (categoria); n_docs = total da categoria"),
        "tsne_theta": _SEM_CSV,
        "grid_k": (("*metrics.csv",),
                   "k_grid_scores = C_v, k_npmi_scores = NPMI (decide), "
                   "k_diversity_scores = Diversity, k_perplexity_scores = "
                   "perplexidade — todos por K"),
        "grid_priors": (("*_grid.csv",),
                        "cv e npmi (npmi decide, cv só reporta) por combinação de "
                        "alpha × eta"),
    },
    "nmf": {
        "keywords_barchart": (("*topics_for_eval.csv", "*_beta.csv"),
                              "beta = pesos H (aditivos, NÃO probabilidades)"),
        "heatmap_phi": (("*_beta.csv",),
                        "beta = matriz H tópico×palavra, pesos aditivos não-negativos"),
        "similaridade_topicos": _SEM_CSV,
        "docs_por_topico": (("*exclusividade_ranking.csv", "*metrics.csv",
                             "derived_topic_ranking.csv"), ""),
        "topico_categoria": (("derived_topic_category.csv",),
                             "valores em % da LINHA (categoria)"),
        "tsne_theta": _SEM_CSV,
        "grid_k": (("*metrics.csv",),
                   "k_grid_scores = C_v, k_npmi_scores = NPMI (decide), "
                   "k_diversity_scores = Diversity — todos por K (sem perplexidade "
                   "no NMF)"),
        "grid_hparams": (("*_grid.csv",),
                         "cv, NPMI (decide) e Diversity por kappa × minimum_probability; "
                         "cobertura = porta condicional (protocolo §1.6, vácua se ~100% "
                         "em toda a grade); n_docs_bow_vazio = documentos sem sinal, "
                         "excluídos do denominador de cobertura, quando a coluna existir"),
    },
    "stm": {
        "prevalencia_covariaveis": (("derived_stm_prevalence_significativos.csv",),
                                    "já filtrado por FDR (p_ajustado_bh < 0,05) — o "
                                    "limiar é do pipeline, não um achado seu"),
        "prevalencia_temporal": _SEM_CSV,   # curva suavizada s(date), só no PNG
        "topics_over_time": (("derived_topic_timeline.csv",),
                             "contagem MENSAL observada de docs por tópico"),
        "frex": (("*topics_frex.csv",), "FREX vs. maior probabilidade por tópico"),
        "heatmap_phi": (("*_beta.csv",), "beta = exp(β), matriz tópico×palavra"),
        "similaridade_topicos": _SEM_CSV,
        "topico_categoria": (("derived_topic_category.csv",),
                             "valores em % da LINHA (categoria)"),
        "grid_k_diagnosticos": (("*grid_k_diagnostics.csv", "*metrics.csv"),
                                "exclusivity_stm está em escala ~8-10 (FREX médio), NÃO "
                                "é a exclusividade c-TF-IDF 0-1 das outras tabelas"),
    },
    "bertopic": {
        "topic_map": _SEM_CSV,
        "hierarquia": _SEM_CSV,
        "keywords": (("*topics_for_eval.csv",), "keywords c-TF-IDF por tópico"),
        "sankey_reduce": _SEM_CSV,
        "documents_umap": _SEM_CSV,
        "similaridade_topicos": _SEM_CSV,
        "docs_por_topico": (("*metrics.csv", "*exclusividade_ranking.csv",
                             "derived_topic_ranking.csv"),
                            "outlier_rate/coverage = fração de docs SEM tópico (id -1)"),
        "topico_categoria": (("*categoria_topico.csv", "derived_topic_category.csv"),
                             "o CSV nativo traz CONTAGENS brutas, não percentuais"),
        "evolucao_temporal": (("derived_topic_timeline.csv",),
                              "contagem MENSAL observada de docs por tópico"),
        "macro_temas": (("*macro_temas.csv", "*macro_k_sweep.csv"), ""),
        "robustez_topn": (("*robustness*.csv",),
                          "top_n = nº de PALAVRAS avaliadas por tópico, com K FIXO. "
                          "NÃO é número de tópicos: nada aqui fala sobre granularidade "
                          "ou fragmentação do modelo"),
    },
    "comparacao": {
        "metricas_comparadas": (("comparison_metrics.csv",),
                                "uma linha por modelo × corpus"),
        "trade_offs": (("comparison_metrics.csv",),
                       "coverage = fração de docs com tópico; exclusivity_comparable=False "
                       "marca célula não comparável"),
        "leitura_por_corpus": (("comparison_metrics.csv",), ""),
    },
}


def _br(x, casas=1):
    """Número em convenção PT-BR (vírgula decimal) — o contrato exige isso na
    prosa, e o modelo copia o que recebe: entregar '23.4' convida ao erro."""
    if isinstance(x, float) and x == int(x) and casas == 0:
        x = int(x)
    return f"{x:,.{casas}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _fatos_derivados(slug: str, model: str, csvs: dict[str, str]) -> str:
    """Fatos COMPARATIVOS já calculados em pandas, para o LLM copiar.

    Existe porque a auditoria de 2026-08-01 mostrou que o assessor acerta os
    valores individuais e erra a RELAÇÃO entre eles: superlativo ("a mais
    concentrada"), ordinal ("o 3º maior"), mínimo de uma coluna, e o pareamento
    id↔valor em tabela larga. Regra de prompt não resolveu — pedir "percorra a
    coluna inteira antes de nomear a posição" ainda deixa a varredura com o
    modelo. Aqui a varredura sai do modelo e vira lookup: ele copia uma linha
    pronta em vez de comparar 24 colunas de cabeça.

    Calcula a partir do MESMO texto de snippet que vai ao LLM (não do arquivo em
    disco) — assim o fato nunca contradiz a tabela visível, mesmo se o snippet
    tiver sido truncado.

    Devolve "" quando não sabe derivar nada para o slug: bloco ausente é melhor
    que bloco errado.
    """
    import io

    import pandas as pd

    def ler(*padroes):
        import fnmatch
        for nome in sorted(csvs):
            if any(fnmatch.fnmatch(nome, p) for p in padroes):
                try:
                    df = pd.read_csv(io.StringIO(csvs[nome]))
                    if not df.empty:
                        return df
                except Exception:
                    continue
        return None

    linhas: list[str] = []
    try:
        if slug == "topico_categoria":
            df = ler("derived_topic_category.csv")
            cols = [c for c in (df.columns if df is not None else []) if c.startswith("T")]
            rot = "category" if df is not None and "category" in df.columns else None
            if df is None or not cols or not rot:
                return ""
            m_ = df[cols].max(axis=1)
            # Concentração = maior fatia da LINHA. A linha de MAIOR fatia é a mais
            # concentrada; a de MENOR fatia máxima é a mais dispersa (foi aqui que
            # o modelo inverteu o sentido em lda-folha e stm-tweets).
            i_con, i_dis = m_.idxmax(), m_.idxmin()
            linhas.append(f"- mais CONCENTRADA: {df.loc[i_con, rot]} "
                          f"({_br(m_[i_con])}% em {df[cols].idxmax(axis=1)[i_con]})")
            linhas.append(f"- mais DISPERSA: {df.loc[i_dis, rot]} "
                          f"({_br(m_[i_dis])}% na maior fatia, em "
                          f"{df[cols].idxmax(axis=1)[i_dis]})")
            ordem = " > ".join(f"{df.loc[i, rot]} {_br(m_[i])}%"
                               for i in m_.sort_values(ascending=False).index)
            linhas.append(f"- ordem por concentração: {ordem}")
            linhas.append("- top-3 de cada linha (pares id↔valor já conferidos):")
            for i in df.index:
                top3 = df.loc[i, cols].astype(float).sort_values(ascending=False).head(3)
                pares = " | ".join(f"{k}={_br(v)}%" for k, v in top3.items())
                linhas.append(f"    {df.loc[i, rot]} → {pares}")

        elif slug in ("docs_por_topico", "pyldavis"):
            df = ler("*exclusividade_ranking.csv", "derived_topic_ranking.csv")
            if df is None or "topic_id" not in df.columns:
                return ""
            nome = "topic_name" if "topic_name" in df.columns else None
            for col, rot, casas in (("n_docs", "por n_docs (volume)", 0),
                                    ("exclusividade", "por exclusividade", 3)):
                if col not in df.columns:
                    continue
                s = df.sort_values(col, ascending=False)
                itens = " > ".join(
                    f"T{int(r.topic_id)}={_br(getattr(r, col), casas)}"
                    for r in s.head(6).itertuples())
                linhas.append(f"- ordem {rot}: {itens}")
                topo, base = s.iloc[0], s.iloc[-1]
                alvo = f" ({topo[nome]})" if nome else ""
                linhas.append(f"    maior: T{int(topo.topic_id)}{alvo} = "
                              f"{_br(topo[col], casas)}; menor: T{int(base.topic_id)} = "
                              f"{_br(base[col], casas)}")

        elif slug in ("grid_priors", "grid_hparams"):
            df = ler("*_grid.csv")
            if df is None:
                return ""
            chaves = [c for c in df.columns if c not in ("cv", "perplexity",
                                                         "k", "corpus_version")]
            for col, melhor in (("cv", "maior"), ("perplexity", "menor")):
                if col not in df.columns:
                    continue
                i = df[col].idxmax() if melhor == "maior" else df[col].idxmin()
                cfg = ", ".join(f"{c}={df.loc[i, c]}" for c in chaves)
                casas = 4 if col == "cv" else 2
                linhas.append(f"- {melhor} {col} do grid: {_br(df.loc[i, col], casas)} "
                              f"em [{cfg}]")
            # NPMI decide (protocolo pos-22/08/2026); casing diverge entre arquivos
            # (NMF grava "NPMI", LDA alpha/eta grava "npmi") -- tolerar os dois sem
            # normalizar (nao duplicar a normalizacao que ja vive em _selecao.py).
            col_npmi = ("NPMI" if "NPMI" in df.columns
                        else "npmi" if "npmi" in df.columns else None)
            if col_npmi:
                i = df[col_npmi].idxmax()
                cfg = ", ".join(f"{c}={df.loc[i, c]}" for c in chaves if c != col_npmi)
                linhas.append(f"- maior NPMI do grid: {_br(df.loc[i, col_npmi], 4)} "
                              f"em [{cfg}]")
            # Cobertura (braco NMF, protocolo §1.6): porta de admissibilidade
            # CONDICIONAL -- so decide se o minimo da grade cair abaixo de ~100%.
            # Limiar 0,999 = a MESMA constante `tolerancia` que
            # `_selecao.py::_porta_cobertura_nmf` usa para essa decisao binaria; nao
            # introduzir um limiar novo, para a prosa nunca discordar do pipeline.
            if "cobertura" in df.columns:
                minima, i_min = df["cobertura"].min(), df["cobertura"].idxmin()
                cfg_min = ", ".join(f"{c}={df.loc[i_min, c]}" for c in chaves
                                    if c != "cobertura")
                if minima < 0.999:
                    linhas.append(
                        f"- cobertura NÃO é ~100% em toda a grade: mínimo "
                        f"{_br(minima, 4)} em [{cfg_min}] (máximo "
                        f"{_br(df['cobertura'].max(), 4)}) — porta ATIVA (protocolo "
                        f"§1.6): pode ter decidido a escolha final.")
                else:
                    linhas.append(
                        f"- cobertura fica ~100% em toda a grade (mínimo "
                        f"{_br(minima, 4)}) — porta VÁCUA (protocolo §1.6): não "
                        f"decide aqui, é só reporte.")
            if "n_docs_bow_vazio" in df.columns:
                n_bow_vazio = int(df["n_docs_bow_vazio"].iloc[0])
                if n_bow_vazio:
                    linhas.append(
                        f"- {n_bow_vazio} documento(s) com BOW vazio excluído(s) do "
                        f"denominador de cobertura — não é descarte por "
                        f"kappa/minimum_probability, é documento sem sinal algum.")

        elif slug == "grid_k_diagnosticos":
            df = ler("*grid_k_diagnostics.csv")
            if df is None or "k" not in df.columns:
                return ""
            # Direção de "melhor" por coluna: C_v e exclusividade sobem;
            # semantic_coherence e heldout são negativos (melhor = mais perto de
            # zero); dispersão de resíduo desce. O modelo errou justamente isso.
            direcao = {"cv": "max", "exclusivity_stm": "max",
                       "semantic_coherence": "max", "heldout_likelihood": "max",
                       "residual_dispersion": "min"}
            for col, d in direcao.items():
                if col not in df.columns:
                    continue
                i = df[col].idxmax() if d == "max" else df[col].idxmin()
                extra = " (menos negativo)" if df[col].max() <= 0 else ""
                linhas.append(f"- melhor {col}{extra}: {_br(df.loc[i, col], 3)} "
                              f"em K={int(df.loc[i, 'k'])} "
                              f"(faixa {_br(df[col].min(), 3)} a {_br(df[col].max(), 3)})")
            if {"cv", "semantic_coherence"} <= set(df.columns):
                linhas.append("- ATENÇÃO: cv e semantic_coherence são métricas "
                              "DIFERENTES e divergem neste grid — não trate uma "
                              "como sinônimo da outra.")

        elif slug == "grid_k":
            # A evidência é `k_grid_scores`: um dict JSON DENTRO de uma célula de
            # CSV — o formato mais hostil do conjunto. Sem estes fatos, o modelo
            # descreveu a curva como "aumento progressivo até K=25" ignorando os
            # vales (auditoria 2026-08-01, investigacao-nmf/grid_k).
            import json as _json
            df = ler("*metrics.csv")
            if df is None or df.empty:
                return ""
            melhor_k_por_col: dict[str, int] = {}
            # NPMI e Diversity primeiro (decidem o K no protocolo pos-22/08/2026,
            # Pareto NPMI x Diversity), C_v e perplexidade depois (reportam, nunca
            # decidem) -- ordem deliberada p/ o LLM nao presumir importancia invertida.
            for col, rotulo, maior_e_melhor in (
                    ("k_npmi_scores", "NPMI", True),
                    ("k_diversity_scores", "Diversity", True),
                    ("k_grid_scores", "C_v", True),
                    ("k_perplexity_scores", "perplexidade", False)):
                if col not in df.columns:
                    continue
                try:
                    serie = _json.loads(str(df.iloc[0][col]))
                    pontos = sorted((int(k), float(v)) for k, v in serie.items())
                except Exception:
                    continue
                if not pontos:
                    continue
                melhor = (max if maior_e_melhor else min)(pontos, key=lambda p: p[1])
                melhor_k_por_col[col] = melhor[0]
                valores = [v for _k, v in pontos]
                linhas.append(
                    f"- melhor {rotulo} do grid: {_br(melhor[1], 4)} em K={melhor[0]} "
                    f"(faixa {_br(min(valores), 4)} a {_br(max(valores), 4)})")
                linhas.append(f"- {rotulo} por K: " +
                              " | ".join(f"K={k}: {_br(v, 4)}" for k, v in pontos))
                quedas = [(pontos[i], pontos[i + 1]) for i in range(len(pontos) - 1)
                          if pontos[i + 1][1] < pontos[i][1]]
                if quedas:
                    det = "; ".join(f"K={a[0]} ({_br(a[1], 4)}) para K={b[0]} "
                                    f"({_br(b[1], 4)})" for a, b in quedas)
                    linhas.append(f"- a série de {rotulo} NÃO é monotônica: cai em "
                                  f"{det}. NÃO a descreva como crescimento "
                                  "progressivo, consistente ou monotônico.")
                else:
                    linhas.append(f"- a série de {rotulo} é monotônica neste grid.")
            if ("k_npmi_scores" in melhor_k_por_col and "k_grid_scores" in melhor_k_por_col
                    and melhor_k_por_col["k_npmi_scores"] != melhor_k_por_col["k_grid_scores"]):
                linhas.append(
                    "- ATENÇÃO: o K de melhor NPMI diverge do K de melhor C_v — "
                    "NPMI decide o protocolo (Pareto NPMI×Diversity), C_v é reporte; "
                    "não descreva o K de C_v como o vencedor.")

        elif slug in ("heatmap_phi", "keywords", "keywords_barchart", "frex"):
            # Dois formatos de evidência para o mesmo tipo de figura: matriz
            # tópico×palavra (NMF/STM, *_beta.csv) ou lista de top keywords
            # (LDA/BERTopic, *topics_for_eval.csv). Os 4 erros comparativos da
            # auditoria 2026-08-01 caíram todos aqui, por não haver fato pronto.
            beta = ler("*_beta.csv")
            if beta is not None and "topic_id" in beta.columns:
                num = beta.set_index("topic_id").select_dtypes("number")
                if num.empty:
                    return ""
                linhas.append("- top-3 termos de cada tópico (maior peso na LINHA):")
                for t in num.index:
                    top = num.loc[t].sort_values(ascending=False).head(3)
                    linhas.append(f"    T{t}: " + " | ".join(
                        f"{c}={_br(v, 4)}" for c, v in top.items()))
                linhas.append("- tópico de maior peso de cada termo (COLUNA):")
                for c in num.columns:
                    linhas.append(f"    {c}: maior em T{num[c].idxmax()} "
                                  f"({_br(num[c].max(), 4)})")
                pilha = num.stack().sort_values(ascending=False)
                if len(pilha) >= 2:
                    (t1, w1), (t2, w2) = pilha.index[0], pilha.index[1]
                    linhas.append(f"- maior peso da tabela: T{t1}x{w1}="
                                  f"{_br(pilha.iloc[0], 4)}; 2º maior: T{t2}x{w2}="
                                  f"{_br(pilha.iloc[1], 4)}")
                linhas.append("- a tabela traz só as colunas de maior peso total no "
                              "modelo inteiro, não o vocabulário completo: um termo "
                              "fora dela pode ter peso alto num tópico específico. "
                              "NÃO afirme peso baixo nem ausência para termo que não "
                              "está na tabela.")
            else:
                kw = ler("*topics_for_eval.csv", "*topics_frex.csv")
                col = next((c for c in ("keywords", "keywords_frex")
                            if kw is not None and c in kw.columns), None)
                if col is None:
                    return ""
                # Aviso explícito quando a evidência NÃO tem coluna numérica: sem
                # ele o modelo inventou peso ("bolsonaro 0,761") a partir de um
                # CSV que só tem id, nome e lista de termos (auditoria de
                # 2026-08-01, 2ª medição, stm-tweets/frex).
                if kw.select_dtypes("number").drop(columns=["topic_id"],
                                                   errors="ignore").empty:
                    linhas.append(
                        "- esta evidência NÃO tem nenhuma coluna numérica: são só "
                        "id, nome e lista de termos por tópico. É PROIBIDO citar "
                        "peso, probabilidade, score ou FREX numérico de qualquer "
                        "termo — esses valores não existem aqui.")
                mapa = {r["topic_id"]: [t.strip() for t in
                                        str(r[col]).split(",") if t.strip()]
                        for _i, r in kw.iterrows()}
                ids = sorted(mapa)
                pares = []
                for i, a in enumerate(ids):
                    for b in ids[i + 1:]:
                        comuns = sorted(set(mapa[a]) & set(mapa[b]))
                        if comuns:
                            pares.append((a, b, comuns))
                if pares:
                    linhas.append("- pares de tópicos que COMPARTILHAM keyword(s):")
                    for a, b, comuns in pares[:40]:
                        linhas.append(f"    T{a} e T{b}: {', '.join(comuns)}")
                    if len(pares) > 40:
                        linhas.append(f"    (+{len(pares) - 40} pares omitidos)")
                else:
                    linhas.append("- nenhum par de tópicos compartilha keyword nesta lista.")
                from collections import Counter
                cont = Counter(t for ks in mapa.values() for t in set(ks))
                # Contagem POR TERMO: sem ela o modelo estimou de cabeça quantos
                # tópicos contêm um termo e errou ("bolsonaro em pelo menos 15
                # tópicos"; são 12) — auditoria de 2026-08-01, 2ª medição.
                repetidos = sorted(((n, t) for t, n in cont.items() if n > 1),
                                   reverse=True)
                if repetidos:
                    linhas.append("- em quantos tópicos desta lista cada termo "
                                  "repetido aparece:")
                    for n, t in repetidos[:25]:
                        onde = [f"T{i}" for i in sorted(mapa) if t in mapa[i]]
                        linhas.append(f"    {t}: {n} tópicos ({', '.join(onde)})")
                    if len(repetidos) > 25:
                        linhas.append(f"    (+{len(repetidos) - 25} termos repetidos "
                                      "omitidos — não estime a contagem deles)")
                unicos = sorted(t for t, n in cont.items() if n == 1)
                linhas.append(
                    f"- {len(unicos)} termos aparecem no top-N de um ÚNICO tópico "
                    "desta lista. Isso NÃO os torna exclusivos desse tópico no "
                    "modelo: a lista é o top-N por tópico, não a distribuição "
                    "completa. Proibido escrever 'exclusivo', 'único' ou 'só "
                    "aparece em' para um termo.")

        elif slug in ("evolucao_temporal", "topics_over_time"):
            df = ler("derived_topic_timeline.csv")
            cols = [c for c in (df.columns if df is not None else []) if c.startswith("T")]
            mes = "mes" if df is not None and "mes" in df.columns else None
            if df is None or not cols or not mes:
                return ""
            linhas.append("- pico de cada tópico (mês, valor):")
            for c in cols:
                i = df[c].idxmax()
                if df.loc[i, c] > 0:
                    linhas.append(f"    {c}: {int(df.loc[i, c])} em {df.loc[i, mes]}")
            linhas.append("- tópico dominante de cada mês:")
            for i in df.index:
                s = df.loc[i, cols].astype(float)
                linhas.append(f"    {df.loc[i, mes]}: {s.idxmax()}={int(s.max())}")

        elif slug == "prevalencia_covariaveis":
            df = ler("derived_stm_prevalence_significativos.csv")
            if df is None or "estimate" not in df.columns:
                return ""
            for rot, sub in (("positivos", df.nlargest(5, "estimate")),
                             ("negativos", df.nsmallest(5, "estimate"))):
                itens = " | ".join(f"T{int(r.topic_id)}×{r.term}={_br(r.estimate, 3)}"
                                   for r in sub.itertuples())
                linhas.append(f"- 5 maiores efeitos {rot}: {itens}")

        elif slug in ("metricas_comparadas", "trade_offs", "leitura_por_corpus"):
            df = ler("comparison_metrics.csv")
            if df is None:
                return ""
            chave = [c for c in ("model", "modelo") if c in df.columns]
            chave += [c for c in ("corpus",) if c in df.columns]
            if not chave:
                return ""
            for col in ("cv", "c_v", "coherence_cv", "diversity", "exclusivity",
                        "coverage"):
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                i, j = df[col].idxmax(), df[col].idxmin()
                nom = lambda k: " / ".join(str(df.loc[k, c]) for c in chave)  # noqa: E731
                linhas.append(f"- {col}: maior = {nom(i)} ({_br(df.loc[i, col], 3)}); "
                              f"menor = {nom(j)} ({_br(df.loc[j, col], 3)})")
    except Exception:
        return ""   # fato duvidoso é pior que fato ausente

    if not linhas:
        return ""
    return ("FATOS DERIVADOS (já calculados a partir do CSV abaixo — copie-os; "
            "NÃO recalcule nem reordene):\n" + "\n".join(linhas))


def evidencia_por_slug(evidence, model_type):
    """Separa os CSVs da evidência por slug. Retorna (com_dados, sem_dados).

    `com_dados`: [(slug, titulo, texto_da_evidencia)] — só slugs cujo CSV existe
    de fato nesta rodada. `sem_dados`: [slug] — recebem o placeholder sem gastar
    chamada de LLM.
    """
    import fnmatch
    m = (model_type or "").lower()
    mapa = VIZ_EVIDENCE.get(m, {})
    snippets = evidence.get("csv_snippets", {})
    com, sem = [], []
    for slug, titulo in VIZ_SLUGS[m]:
        padroes, glossario = mapa.get(slug, _SEM_CSV)
        casados = [n for n in snippets
                   if any(fnmatch.fnmatch(n, p) for p in padroes)]
        if not casados:
            sem.append(slug)
            continue
        blocos = [f"#### {n}\n```csv\n{snippets[n]}```" for n in sorted(casados)]
        texto = "\n".join(blocos)
        # Fatos derivados ANTES da tabela: é o que o modelo deve citar. A tabela
        # crua fica para os valores de célula que a prosa também usa.
        fatos = _fatos_derivados(slug, m, {n: snippets[n] for n in casados})
        if fatos:
            texto = f"{fatos}\n{texto}"
        if glossario:
            texto = f"_Leitura das colunas: {glossario}._\n{texto}"
        com.append((slug, titulo, texto))
    return com, sem


def build_viz_audit_messages(evidence, model_type, *, prompts=None):
    """Mensagens da fase site: [system, user]. Sem histórico e sem params_patch.

    A evidência vai AGRUPADA POR SLUG, não num bloco único: assim o modelo não
    precisa adivinhar qual CSV sustenta qual visualização (era a origem dos
    erros publicados). Slugs sem CSV não entram na lista pedida.
    """
    prompts = prompts or load_advisor_prompts()
    m = (model_type or "").lower()
    method = prompts["methods"].get(m)
    if not method or "site" not in method:
        raise AdvisorConfigError(f"fase 'site' ausente para o modelo {m!r}")
    shared = prompts["shared"]
    com, _sem = evidencia_por_slug(evidence, m)
    if com:
        viz_list = "\n".join(f"  - {slug}: {titulo}" for slug, titulo, _ in com)
        snippets = "\n\n".join(f"### {slug}\n{texto}" for slug, _, texto in com)
    else:   # nenhuma viz desta rodada tem CSV — mantém o formato, sem inventar
        viz_list = "  (nenhuma visualização com dados nesta rodada)"
        snippets = "(sem CSVs)"
    tokens = {"CORPUS": evidence.get("corpus", "?"),
              "CORPUS_INFO": evidence.get("corpus_info", ""),
              "VIZ_LIST": viz_list,
              "VIZ_AUDIT_CONTRACT": shared["viz_audit_contract"],
              "EVIDENCE": snippets}
    user_msg = {"role": "user", "content": _fill(method["site"], tokens)}
    # `system_site` quando existir: o `system` foi escrito para a calibração de
    # hiperparâmetros e mandava "propor a próxima config" — o oposto desta fase.
    system_txt = method.get("system_site") or method["system"]
    system_msg = {"role": "system",
                  "content": _fill(system_txt, {"EVAL_METRICS": shared["eval_metrics"]})}
    return [system_msg, user_msg]


def _slug_do_cabecalho(linha: str) -> str | None:
    """Extrai o slug de uma linha de cabeçalho, tolerando o que o LLM varia.

    O contrato pede '## <slug>' exato, mas a aderência varia por modelo. Antes,
    qualquer desvio fazia a linha virar CORPO da seção anterior — contaminando o
    arquivo anterior e marcando o slug como faltante. Aceita: '### slug',
    '## slug — Título', '## 1. slug', '## **slug**', '## Slug', '## slug phi'.
    """
    m = re.match(r"^#{2,3}\s+(.*?)\s*$", linha)
    if not m:
        return None
    bruto = re.sub(r"^\d+[.)]\s*", "", m.group(1))       # numeração
    bruto = re.split(r"\s+[—–:]\s*|\s+-\s+", bruto)[0]   # título residual
    bruto = bruto.strip("*` ").strip()                    # ênfase markdown
    cand = re.sub(r"[\s\-]+", "_", bruto.lower())
    return cand or None


def parse_viz_audit(response: str, expected_slugs: list[str]) -> tuple[dict[str, str], list[str]]:
    """Divide a resposta em seções '## <slug>'. Cabeçalhos fora da lista são
    descartados (o LLM às vezes inventa seções). Retorna (sections, missing)."""
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in (response or "").splitlines():
        cand = _slug_do_cabecalho(line.strip())
        if cand is not None:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = cand if cand in expected_slugs else None
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    sections = {s: t for s, t in sections.items() if t}
    missing = [s for s in expected_slugs if s not in sections]
    return sections, missing


def sanitize_viz_markdown(text: str) -> str:
    """Neutraliza construções que quebram o render do Quarto.

    O texto vem de um LLM e é incluído via {{< include >}} DENTRO de um callout
    `:::`. Já houve uma quebra de build real (`---` lido pelo Pandoc como bloco
    de metadados YAML no meio do documento). Vetores tratados:
      - bloco <think>...</think> (modelos com raciocínio explícito — o fallback
        configurado é dessa família);
      - linha `---`/`:::` no corpo (fecha o callout ou vira YAML);
      - cerca ``` ímpar (engole o resto da página);
      - shortcode `{{< ... >}}` (o Quarto tenta resolver).
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*", "", text, flags=re.S | re.I)   # bloco não fechado
    linhas = []
    for linha in text.splitlines():
        despido = linha.strip()
        if re.fullmatch(r"-{3,}", despido):
            linha = linha.replace("-", "*")        # separador seguro
        elif despido.startswith(":::"):
            linha = linha.replace(":::", "∶∶∶", 1)  # não fecha o callout
        linhas.append(linha)
    text = "\n".join(linhas)
    if text.count("```") % 2:
        text = text.rstrip() + "\n```"
    text = text.replace("{{<", "{{ <")
    return text.strip()


MARCA_NOTA_AUTOR = "> **Nota do autor"
ROTULOS_LEITURA = ["A", "B", "C", "D"]


def _nota_do_autor(path: str) -> str:
    """Nota de auditoria humana já gravada no arquivo, para sobreviver ao re-run.

    Os textos publicados passam por auditoria contra os CSVs e podem receber uma
    ressalva ('> **Nota do autor ...') apontando erro de uma das leituras. Sem
    reinjetá-la, a próxima execução da fase site republica o callout SEM a
    ressalva — o .bak preserva o arquivo, mas quem lê o site não vê nada.
    """
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        texto = f.read()
    # rsplit: o rodapé é sempre o ÚLTIMO '***'. Com split() na PRIMEIRA
    # ocorrência, um '---' na prosa do LLM — que `sanitize_viz_markdown`
    # converte em '***' — cortava o corpo ANTES da nota, e a ressalva sumia
    # silenciosamente no re-run (o .bak guardava, mas o site ia ao ar sem ela).
    corpo = texto.rsplit("\n***\n", 1)[0]
    linhas = corpo.splitlines()
    for i, linha in enumerate(linhas):
        if not linha.startswith(MARCA_NOTA_AUTOR):
            continue
        # A nota pode ter linhas de continuação (parágrafo de auditoria);
        # devolver só a primeira descartava o resto no re-run.
        bloco = [linha]
        for proxima in linhas[i + 1:]:
            if not proxima.strip():
                break
            bloco.append(proxima)
        return "\n".join(bloco)
    return ""


def _rodape_viz(quem: list[str], plural: bool) -> str:
    """Rodapé de proveniência. `quem` = 'provider/model' de quem respondeu.

    Separador '***' (nao '---'): o Pandoc interpreta uma linha '---' seguida de
    texto como bloco de metadados YAML no MEIO do documento, e o site quebra ao
    dar {{< include >}} neste arquivo dentro de um callout.
    """
    import datetime
    hoje = datetime.date.today().isoformat()
    fonte = f"({' e '.join(quem)}, {hoje})"
    aviso = ("os valores individuais costumam estar corretos, mas comparações, "
             "ordenações e superlativos exigem conferência contra os dados.")
    if plural:
        return (f"\n\n***\n*Duas leituras independentes geradas por IA a partir "
                f"dos mesmos CSVs do run {fonte}, com o mesmo prompt e sem "
                f"acesso uma à outra. **Podem conter erros** — {aviso} "
                "Não são conclusão do autor.*\n")
    return (f"\n\n***\n*Texto gerado por IA {fonte} a partir dos CSVs do run. "
            f"**Pode conter erros** — {aviso} Não é conclusão do autor.*\n")


def _gravar_viz(run_dir, corpos: dict, full_response: str) -> list:
    """Escreve os <slug>.md compostos + a resposta íntegra. Retorna os paths.

    Conteúdo diferente de um arquivo já existente NÃO é descartado: o antigo vai
    para <slug>.md.bak-<stamp>. Esses textos passam por revisão humana e vão ao
    site — reexecutar a auditoria não pode apagar a versão revisada.
    """
    import datetime
    out = os.path.join(run_dir, "advisor_viz")
    os.makedirs(out, exist_ok=True)
    written = []
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for slug, corpo in corpos.items():
        path = os.path.join(out, f"{slug}.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                antigo = f.read()
            if antigo != corpo:
                os.replace(path, f"{path}.bak-{stamp}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(corpo)
        written.append(path)
    with open(os.path.join(run_dir, "advisor_viz_report.md"), "w", encoding="utf-8") as f:
        f.write(full_response)
    return written


def persist_viz_audit(run_dir, corpus, model, sections, full_response, advisor_cfg):
    """Fase site com UMA leitura: grava <run_dir>/advisor_viz/<slug>.md com o
    rodapé de proveniência do provider que respondeu.

    Atalho de `persist_viz_panel` para painel de um só integrante — é o caminho
    de quando `advisor.viz_panel` está desligado, ou de quando um dos dois
    providers falhou e só o outro respondeu.
    """
    return persist_viz_panel(run_dir, corpus, model,
                             [(advisor_cfg, sections)], full_response)


def persist_viz_panel(run_dir, corpus, model, leituras, full_response):
    """Fase site em painel: as N leituras vão para o MESMO <slug>.md, rotuladas
    'Leitura A/B', com um rodapé único citando todos os providers.

    `leituras` = [(cfg, sections), ...] na ordem em que devem aparecer. Leituras
    de texto IDÊNTICO viram um bloco só, sem rótulo: é o caso do placeholder de
    slug sem CSV, que é inserido em código e sai igual dos dois providers —
    publicá-lo duas vezes como 'leituras independentes' seria falso.
    """
    slugs = []
    for _cfg, secs in leituras:
        for slug in secs:
            if slug not in slugs:
                slugs.append(slug)
    corpos = {}
    for slug in slugs:
        blocos, quem = [], []
        for cfg, secs in leituras:
            if slug not in secs:
                continue
            texto = sanitize_viz_markdown(secs[slug])
            quem.append(f"{cfg.get('provider')}/{cfg.get('model')}")
            if texto not in blocos:
                blocos.append(texto)
        if not blocos:
            continue
        if len(blocos) == 1:
            corpo = blocos[0]
        else:
            corpo = "\n\n".join(
                f"**Leitura {ROTULOS_LEITURA[i]} — {leituras[i][0].get('model')}**"
                f"\n\n{texto}" for i, texto in enumerate(blocos))
        nota = _nota_do_autor(os.path.join(run_dir, "advisor_viz", f"{slug}.md"))
        if nota:
            corpo = f"{corpo}\n\n{nota}"
        corpos[slug] = corpo + _rodape_viz(quem, plural=len(blocos) > 1)
    return _gravar_viz(run_dir, corpos, full_response)


class AdvisorConfigError(Exception):
    """Raised on missing/disabled advisor config or absent API key."""


_truststore_injected = False


def _use_os_truststore() -> None:
    """Faz o Python (httpx/ssl) confiar no store de certificados do SO.

    Necessário em redes com interceptação SSL (proxy corporativo / CA raiz
    customizada): o Windows já confia na CA do proxy, mas o httpx usa o bundle
    do certifi por padrão e recusa o certificado. `truststore` redireciona o
    ssl para o store do SO. Idempotente; no-op se truststore não estiver
    instalado (aí vale o trust padrão).
    """
    global _truststore_injected
    if _truststore_injected:
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass
    _truststore_injected = True


def _build_client(cfg: dict):
    """Constrói o client OpenAI-SDK para um provider (openai | ollama).

    Ambos usam o SDK `openai`; muda só o base_url e se a chave é exigida.
    Ollama local não usa chave (dummy); Ollama remoto/cloud lê de api_key_env.
    openai lê a chave de os.environ[cfg["api_key_env"]].
    """
    _use_os_truststore()
    provider = cfg.get("provider", "ollama")
    if provider == "ollama":
        # Ollama remoto/cloud é autenticado (lê a chave de api_key_env, ex.:
        # OLLAMA_API_KEY); Ollama local não usa chave — cai na dummy "ollama"
        # (o SDK exige algo não-vazio; o servidor local ignora).
        from openai import OpenAI
        key_env = cfg.get("api_key_env")
        api_key = (os.environ.get(key_env) if key_env else None) or "ollama"
        return OpenAI(
            api_key=api_key,
            base_url=cfg.get("base_url", "http://localhost:11434/v1"),
        )
    if provider == "openai":
        key_env = cfg.get("api_key_env")
        api_key = os.environ.get(key_env) if key_env else None
        if not api_key:
            raise AdvisorConfigError(f"env var {key_env!r} not set (advisor.api_key_env)")
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=cfg.get("base_url"))
    raise AdvisorConfigError(f"unknown provider {provider!r} (use 'openai' or 'ollama')")


# --- Auditoria de gastos ------------------------------------------------------
# Preços em USD por 1M de tokens (developers.openai.com/api/docs/pricing,
# consultado em 2026-08-01). Sobrescreva em params.yaml > advisor.pricing sem
# mexer no código — preço muda, e um número velho aqui vira relatório errado.
# Modelo AUSENTE da tabela custa 0: é o caso do Ollama (local ou cloud), que não
# cobra por token. Zero aqui significa "não tarifado", não "de graça de fato".
_PRECOS_PADRAO: dict[str, dict[str, float]] = {
    "gpt-4o":         {"entrada": 2.50, "cacheada": 1.25,  "saida": 10.00},
    "gpt-4o-mini":    {"entrada": 0.15, "cacheada": 0.075, "saida": 0.60},
    "gpt-4.1":        {"entrada": 2.00, "cacheada": 0.50,  "saida": 8.00},
    "gpt-4.1-mini":   {"entrada": 0.40, "cacheada": 0.10,  "saida": 1.60},
    "gpt-4.1-nano":   {"entrada": 0.10, "cacheada": 0.025, "saida": 0.40},
    "gpt-5":          {"entrada": 1.25, "cacheada": 0.125, "saida": 10.00},
    "gpt-5-mini":     {"entrada": 0.25, "cacheada": 0.025, "saida": 2.00},
    "gpt-5-nano":     {"entrada": 0.05, "cacheada": 0.005, "saida": 0.40},
    "gpt-5.1":        {"entrada": 1.25, "cacheada": 0.125, "saida": 10.00},
    "gpt-5.2":        {"entrada": 1.75, "cacheada": 0.175, "saida": 14.00},
    "gpt-5.4":        {"entrada": 2.50, "cacheada": 0.25,  "saida": 15.00},
    "gpt-5.4-mini":   {"entrada": 0.75, "cacheada": 0.075, "saida": 4.50},
    "gpt-5.4-nano":   {"entrada": 0.20, "cacheada": 0.02,  "saida": 1.25},
    "gpt-5.5":        {"entrada": 5.00, "cacheada": 0.50,  "saida": 30.00},
    "gpt-5.6-luna":   {"entrada": 0.20, "cacheada": 0.02,  "saida": 1.20},
    "gpt-5.6-terra":  {"entrada": 2.00, "cacheada": 0.20,  "saida": 12.00},
    "gpt-5.6-sol":    {"entrada": 5.00, "cacheada": 0.50,  "saida": 30.00},
    "o3-mini":        {"entrada": 1.10, "cacheada": 0.55,  "saida": 4.40},
    "o4-mini":        {"entrada": 1.10, "cacheada": 0.275, "saida": 4.40},
}
_PRECOS: dict[str, dict[str, float]] = dict(_PRECOS_PADRAO)

# Contexto da chamada corrente (corpus / modelo de tópico / fase), preenchido por
# quem orquestra. `_call_one` não tem como saber disso sozinho, e sem esses
# campos o log vira uma lista de números sem dono.
_USO_CTX: dict = {}


def caminho_log_de_uso() -> str:
    """JSONL append-only com uma linha por chamada de LLM.

    Fica em 03-topic-modeling/data/output/ — resolvido a partir da localização
    DESTE arquivo, não do cwd (o advisor roda de notebooks/ e da raiz).

    `ADVISOR_USAGE_LOG` redireciona o destino. Existe porque a suíte de testes
    exercita `_call_one` com client mockado, e sem o redirecionamento cada
    `pytest` sujava o log real com chamadas fantasma (4 tokens, modelo "?"),
    corrompendo o relatório de custo.
    """
    env = os.environ.get("ADVISOR_USAGE_LOG")
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "output", "advisor_usage.jsonl")


def carregar_precos(params: dict | None = None) -> None:
    """Aplica advisor.pricing do params.yaml por cima da tabela padrão."""
    global _PRECOS
    _PRECOS = dict(_PRECOS_PADRAO)
    custom = ((params or {}).get("advisor", {}) or {}).get("pricing") or {}
    for modelo, faixas in custom.items():
        if isinstance(faixas, dict):
            _PRECOS[modelo] = {**_PRECOS.get(modelo, {}), **faixas}


def contar_tokens(texto: str, model: str = "") -> int:
    """Tokens de `texto`. Só é usado quando o provider NÃO devolve `usage`.

    O `usage` da resposta é sempre preferido: é o que a OpenAI de fato cobra,
    já inclui tokens de raciocínio e desconta o que veio de cache. Isto aqui é
    a estimativa para o Ollama, que nem sempre reporta.
    """
    if not texto:
        return 0
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(texto))
    except Exception:
        return max(1, len(texto) // 4)   # sem tiktoken: aproximação grosseira


def custo_estimado(model: str, entrada: int = 0, saida: int = 0,
                   entrada_cacheada: int = 0) -> float:
    """USD da chamada. Modelo fora da tabela de preços (Ollama) devolve 0,0.

    `entrada` é o total de tokens de prompt; `entrada_cacheada` é a parte dele
    que veio de cache (tarifada mais barato) e é DESCONTADA de `entrada`.
    """
    p = _PRECOS.get((model or "").strip())
    if not p:
        return 0.0
    nao_cacheada = max(0, entrada - max(0, entrada_cacheada))
    return round(
        nao_cacheada / 1e6 * p.get("entrada", 0.0)
        + max(0, entrada_cacheada) / 1e6 * p.get("cacheada", p.get("entrada", 0.0))
        + max(0, saida) / 1e6 * p.get("saida", 0.0),
        6)


def registrar_uso(cfg: dict, *, entrada: int, saida: int, entrada_cacheada: int = 0,
                  raciocinio: int = 0, contexto: dict | None = None,
                  path: str | None = None) -> dict:
    """Acrescenta uma linha ao log de uso. Nunca levanta: log é observabilidade,
    não pode derrubar uma rodada de análise que já custou a chamada."""
    import datetime
    ctx = dict(contexto or {})
    model = cfg.get("model", "?")
    linha = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "provider": cfg.get("provider", "?"),
        "model": model,
        "corpus": ctx.get("corpus"),
        "modelo_topico": ctx.get("modelo_topico"),
        "fase": ctx.get("fase"),
        "entrada": int(entrada or 0),
        "entrada_cacheada": int(entrada_cacheada or 0),
        "saida": int(saida or 0),
        "raciocinio": int(raciocinio or 0),
        "custo_usd": custo_estimado(model, entrada or 0, saida or 0,
                                    entrada_cacheada or 0),
        "tarifado": model in _PRECOS,
    }
    try:
        destino = path or caminho_log_de_uso()
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        with open(destino, "a", encoding="utf-8") as f:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return linha


def relatorio_de_gastos(path: str | None = None) -> str:
    """Relatório markdown do log de uso: por rodada, por modelo, por fase, total."""
    destino = path or caminho_log_de_uso()
    if not os.path.exists(destino):
        return (f"# Gastos do assessor\n\nSem log ainda (`{destino}` não existe). "
                "O arquivo é criado na próxima chamada de LLM.\n")
    linhas = []
    with open(destino, encoding="utf-8") as f:
        for ln in f:
            try:
                linhas.append(json.loads(ln))
            except Exception:
                continue
    if not linhas:
        return "# Gastos do assessor\n\nLog vazio.\n"

    def soma(rs, campo):
        return sum(r.get(campo) or 0 for r in rs)

    def bloco(titulo, chave):
        grupos: dict = {}
        for r in linhas:
            grupos.setdefault(chave(r), []).append(r)
        out = [f"\n## {titulo}\n",
               "| | chamadas | entrada | saída | raciocínio | custo (USD) |",
               "|---|---:|---:|---:|---:|---:|"]
        for k in sorted(grupos, key=lambda k: -soma(grupos[k], "custo_usd")):
            rs = grupos[k]
            out.append(f"| {k} | {len(rs)} | {soma(rs,'entrada'):,} | "
                       f"{soma(rs,'saida'):,} | {soma(rs,'raciocinio'):,} | "
                       f"{soma(rs,'custo_usd'):.4f} |")
        return "\n".join(out).replace(",", ".")

    total = soma(linhas, "custo_usd")
    nao_tarifados = [r for r in linhas if not r.get("tarifado")]
    cab = [
        "# Gastos do assessor (LLM)", "",
        f"- Log: `{destino}`",
        f"- Período: {linhas[0]['timestamp']} → {linhas[-1]['timestamp']}",
        f"- Chamadas: **{len(linhas)}**",
        f"- Tokens: {soma(linhas,'entrada'):,} entrada "
        f"({soma(linhas,'entrada_cacheada'):,} de cache) / "
        f"{soma(linhas,'saida'):,} saída "
        f"({soma(linhas,'raciocinio'):,} de raciocínio)".replace(",", "."),
        f"- **TOTAL: US$ {total:.4f}**",
    ]
    if nao_tarifados:
        cab.append(f"- {len(nao_tarifados)} chamada(s) sem preço na tabela "
                   "(Ollama ou modelo novo) contam como 0 — não são gratuitas "
                   "por definição, apenas não tarifadas por token.")
    return "\n".join(cab) + "\n" + "\n".join([
        bloco("Por modelo", lambda r: f"{r.get('provider')}/{r.get('model')}"),
        bloco("Por fase", lambda r: r.get("fase") or "(sem fase)"),
        bloco("Por rodada (corpus × modelo de tópico)",
              lambda r: f"{r.get('corpus')}/{r.get('modelo_topico')}"),
        bloco("Por dia", lambda r: (r.get("timestamp") or "")[:10]),
    ]) + "\n"


def _e_modelo_de_raciocinio(model: str) -> bool:
    """gpt-5*, o1/o3/o4* — família de raciocínio da OpenAI.

    Elas recusam `max_tokens` (exigem `max_completion_tokens`) e recusam
    `temperature` != 1. Enviar os parâmetros antigos dá 400, e o 400 vira
    exceção que `call_advisor` trata como falha do primary: o fallback (Ollama)
    responde e o run inteiro sai do modelo errado, com o rodapé registrando
    corretamente o fallback — mas passa despercebido se ninguém ler o rodapé.
    """
    m = (model or "").lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _call_one(messages: list, cfg: dict) -> str:
    """Uma única chamada de LLM (sem fallback). Recebe a lista de mensagens.

    `timeout_sec`/`max_retries` são configuráveis: os defaults do SDK (600 s de
    leitura × 2 retries) faziam uma chamada travada no primary segurar o
    notebook por ~30 min ANTES de o fallback ser sequer tentado.
    """
    client = _build_client(cfg)
    model = cfg["model"]
    limite = cfg.get("max_output_tokens", 4000)
    kwargs: dict = {"model": model, "messages": messages,
                    "timeout": cfg.get("timeout_sec", 120)}
    if _e_modelo_de_raciocinio(model):
        # O limite cobre tokens de RACIOCÍNIO + resposta. Com o teto de prosa
        # (~4k) o raciocínio consome a cota e a resposta volta VAZIA — folga de
        # 4x. `reasoning_effort` fica configurável: esta fase é cópia de fatos
        # já calculados, não dedução, então "low" costuma bastar.
        kwargs["max_completion_tokens"] = max(limite * 4, 8000)
        if cfg.get("reasoning_effort"):
            kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    else:
        kwargs["max_tokens"] = limite
        kwargs["temperature"] = cfg.get("temperature", 0.2)
    resp = client.chat.completions.create(**kwargs)
    conteudo = resp.choices[0].message.content
    # Log de custo no PONTO da chamada: assim toda rodada entra no relatório sem
    # depender de o orquestrador lembrar de registrar — inclusive as que caíram
    # no fallback, que são justamente as que passam despercebidas.
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            det_in = getattr(u, "prompt_tokens_details", None)
            det_out = getattr(u, "completion_tokens_details", None)
            registrar_uso(
                cfg,
                entrada=getattr(u, "prompt_tokens", 0) or 0,
                saida=getattr(u, "completion_tokens", 0) or 0,
                entrada_cacheada=getattr(det_in, "cached_tokens", 0) or 0,
                raciocinio=getattr(det_out, "reasoning_tokens", 0) or 0,
                contexto=_USO_CTX)
        else:   # provider sem `usage` (Ollama local): estima com tiktoken
            registrar_uso(
                cfg,
                entrada=sum(contar_tokens(m.get("content", ""), model)
                            for m in messages),
                saida=contar_tokens(conteudo or "", model),
                contexto={**_USO_CTX, "estimado": True})
    except Exception:
        pass
    return conteudo


def call_advisor(messages: list, advisor_cfg: dict) -> tuple[str, dict]:
    """Chama o LLM primary; em QUALQUER exceção, tenta o `fallback` (se houver).

    `messages` = lista [{role, content}] (system + histórico + user). Cobre os
    modos de falha do primary (sem chave, rede, rate limit, timeout). O bloco
    `fallback` herda temperature/max_output_tokens do pai quando não os define;
    provider/model/base_url/api_key_env vêm do próprio fallback.

    Devolve (resposta, cfg_efetiva) — Correção 2: sem a cfg efetiva, quem
    persiste a resposta (rodapé de `persist_viz_audit`, entrada do ledger) não
    tem como saber se foi o primary ou o fallback que respondeu, e citava
    sempre `advisor_cfg` (o primary) mesmo quando ele tinha falhado.
    """
    if not advisor_cfg.get("enabled", False):
        raise AdvisorConfigError("advisor.enabled is false in params.yaml")
    try:
        return _call_one(messages, advisor_cfg), advisor_cfg
    except Exception as primary_exc:
        fb = advisor_cfg.get("fallback")
        if not fb:
            raise
        inherited = {k: advisor_cfg[k] for k in ("temperature", "max_output_tokens")
                     if k in advisor_cfg}
        merged = {**inherited, **fb}
        try:
            return _call_one(messages, merged), merged
        except Exception as fb_exc:
            raise AdvisorConfigError(
                f"primary ({advisor_cfg.get('provider')}) e fallback "
                f"({fb.get('provider')}) falharam: {primary_exc!r} / {fb_exc!r}"
            ) from fb_exc


# --- Task 7: persistência da rodada ------------------------------------------
def persist_round(base_output, corpus, model, round_n, evidence, proposal,
                  invalid_keys, report_md, status, cycle_round=None, phase=None,
                  advisor_cfg=None):
    os.makedirs(base_output, exist_ok=True)
    proposal_file = os.path.join(base_output, f"advisor_round_{round_n}_proposal.yaml")
    report_file = os.path.join(base_output, f"advisor_round_{round_n}_report.md")

    with open(proposal_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(proposal, f, allow_unicode=True, sort_keys=False)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md or "")

    ledger_path = os.path.join(base_output, "llm_advisor_state.json")
    entry = {
        "round": round_n,
        "cycle_round": cycle_round,
        "phase": phase,
        "status": status,
        "evidence_source": os.path.basename(str(evidence.get("run_dir", ""))) or evidence.get("run_dir", ""),
        "target_layer": proposal.get("target_layer"),
        "proposal_file": os.path.basename(proposal_file),
        "report_file": os.path.basename(report_file),
        "params_patch": proposal.get("params_patch", {}),
        "invalid_keys": invalid_keys,
    }
    if advisor_cfg is not None:
        # Correção 2: cfg EFETIVAMENTE usada (pode ser o fallback) — sem isso
        # o ledger sempre citaria o primary configurado, mesmo quando ele
        # falhou e quem respondeu foi o fallback.
        entry["provider"] = advisor_cfg.get("provider")
        entry["model"] = advisor_cfg.get("model")
    append_ledger(ledger_path, corpus, model, entry)
    return {"proposal_file": proposal_file, "report_file": report_file, "ledger": ledger_path}


# --- Task 8: orquestrador ----------------------------------------------------
def run_advisor_round(corpus, model, base_output, params=None, round=None,
                      phase=None, retry_on_format_error=True):
    if params is None:
        import _helpers  # lazy: _helpers é pesado (~23s: gensim/spacy) — só carrega
        params = _helpers.load_params()  # em uso real, nunca nos testes (passam params)
    advisor_cfg = params.get("advisor", {})
    rounds = int(advisor_cfg.get("rounds", 3))
    prompts = load_advisor_prompts()

    ledger_path = os.path.join(base_output, "llm_advisor_state.json")
    chat_path = os.path.join(base_output, "advisor_chat.json")
    # round_n = global monotônico (FILE NAMING + ledger, sem colisão de arquivo).
    round_n = resolve_round(ledger_path, override=round)
    history = read_chat(chat_path)
    # cycle_round = posição DENTRO do ciclo (chat reseta a cada 'final'): 1 turno
    # assistant no chat = já houve 1 rodada -> próxima é a 2. A fase vem DAQUI.
    cycle_round = sum(1 for m in history if m.get("role") == "assistant") + 1
    phase = select_phase(cycle_round, rounds=rounds, override=phase)
    corpus_info = (params.get("corpora", {}).get(corpus, {}) or {}).get("description", "")
    evidence = collect_evidence(corpus, model, base_output, corpus_info=corpus_info)
    carregar_precos(params)
    _USO_CTX.update({"corpus": corpus, "modelo_topico": model, "fase": phase})

    messages = build_advisor_messages(evidence, cycle_round, model, phase=phase,
                                      history=history, prompts=prompts, rounds=rounds)

    # cfg_used = cfg EFETIVAMENTE usada (Correção 2) — call_advisor devolve o
    # bloco primary OU fallback, o que tiver de fato respondido; persist_round
    # grava essa cfg no ledger (não a `advisor_cfg` configurada, que pode ser
    # o primary que falhou).
    response, cfg_used = call_advisor(messages, advisor_cfg)
    try:
        proposal, invalid_keys, report_md = parse_proposal(response)
        status = "proposed"
    except ProposalFormatError:
        if retry_on_format_error:
            # O retry usa a resposta malformada como contexto para corrigir, mas
            # esse turno ruim + o lembrete corretivo NÃO entram na memória de chat
            # (ver bloco save_chat abaixo) — senão toda rodada futura reenviaria o
            # próprio lixo do modelo e uma correção já satisfeita.
            strict_user = {"role": "user", "content":
                "IMPORTANTE: sua resposta anterior não continha um bloco "
                "```yaml proposal``` válido com params_patch. Responda AGORA "
                "estritamente no formato especificado."}
            retry_msgs = messages + [{"role": "assistant", "content": response}, strict_user]
            response, cfg_used = call_advisor(retry_msgs, advisor_cfg)
            try:
                proposal, invalid_keys, report_md = parse_proposal(response)
                status = "proposed"
            except ProposalFormatError:
                proposal, invalid_keys, report_md, status = (
                    {"round": round_n, "target_layer": None, "params_patch": {}},
                    [], response, "needs_review")
        else:
            proposal, invalid_keys, report_md, status = (
                {"round": round_n, "target_layer": None, "params_patch": {}},
                [], response, "needs_review")

    # Memória de chat: só persiste um turno LIMPO quando há proposta válida
    # (`response` = a resposta que parseou, seja a 1ª ou a do retry). Em
    # needs_review NÃO cresce a memória — evita poluir o replay das próximas
    # rodadas com resposta malformada + lembrete corretivo.
    if status == "proposed":
        save_chat(chat_path, corpus, model,
                  messages + [{"role": "assistant", "content": response}])
        # Fim de ciclo: a rodada 'final' (relatório) fecha o ciclo — reseta o chat
        # para a próxima chamada recomeçar limpo na rodada 1/varredura.
        if phase == "final":
            save_chat(chat_path, corpus, model, [])
    paths = persist_round(base_output, corpus, model, round_n, evidence,
                          proposal, invalid_keys, report_md, status,
                          cycle_round=cycle_round, phase=phase,
                          advisor_cfg=cfg_used)
    return {"round": round_n, "cycle_round": cycle_round, "phase": phase,
            "proposal": proposal, "invalid_keys": invalid_keys,
            "report_md": report_md, "paths": paths, "status": status}


def painel_de_leituras(advisor_cfg: dict) -> list[dict]:
    """Cfgs a consultar na fase site. Com `viz_panel`, primary E fallback.

    Cada cfg sai SEM a chave `fallback`: em painel, a falha de um provider não
    pode virar uma segunda chamada ao outro — as duas leituras sairiam do MESMO
    modelo e a página as publicaria como independentes. Sem `viz_panel` o
    comportamento é o antigo (uma cfg, com fallback automático dentro dela).
    """
    fb = advisor_cfg.get("fallback")
    if not (advisor_cfg.get("viz_panel") and fb):
        return [advisor_cfg]
    herdado = {k: advisor_cfg[k] for k
               in ("enabled", "temperature", "max_output_tokens", "timeout_sec",
                   "max_retries", "reasoning_effort")
               if k in advisor_cfg}
    return [{k: v for k, v in advisor_cfg.items() if k != "fallback"},
            {**herdado, **{k: v for k, v in fb.items() if k != "fallback"}}]


def _uma_leitura(messages, cfg, expected, retry_on_missing=True):
    """Uma chamada (+ retry das seções faltantes) contra UM provider."""
    # `call_advisor` devolve (texto, cfg_efetiva): a cfg de quem REALMENTE
    # respondeu. É ela que vai ao rodapé publicado e ao ledger — atribuir ao
    # primary configurado publicaria autoria errada.
    response, cfg_used = call_advisor(messages, cfg)
    response = response or ""
    sections, missing = parse_viz_audit(response, expected)
    cfg_retry = cfg_used
    if missing and retry_on_missing:
        retry_msgs = messages + [
            {"role": "assistant", "content": response},
            {"role": "user", "content":
                "Faltaram as seções: " + ", ".join(missing) +
                ". Responda SOMENTE com essas seções. Formato OBRIGATÓRIO por "
                "seção: uma linha contendo EXATAMENTE '## <slug>' (dois '#', um "
                "espaço, o slug em minúsculas com underscore, NADA mais na "
                "linha — sem título, sem '###', sem numeração), seguida da "
                "prosa. Exemplo:\n## heatmap_phi\n<prosa>"}]
        response2, cfg_retry = call_advisor(retry_msgs, cfg)
        response2 = response2 or ""
        more, _ = parse_viz_audit(response2, missing)
        sections.update(more)
        response = response + "\n\n" + response2
        missing = [s for s in expected if s not in sections]
    return sections, missing, response, cfg_used, cfg_retry


def run_viz_audit(corpus, model, base_output, params=None, retry_on_missing=True):
    """Fase site: grava advisor_viz/<slug>.md no run primário. Prosa pura —
    nunca toca params.yaml nem chat memory.

    Com `advisor.viz_panel: true`, consulta os DOIS providers na mesma rodada e
    publica as duas leituras no mesmo arquivo. Se um deles falhar, o arquivo sai
    com a leitura de quem respondeu (status `partial_panel` no ledger); só se os
    dois falharem a rodada inteira levanta.
    """
    if params is None:
        import _helpers
        params = _helpers.load_params()
    advisor_cfg = params.get("advisor", {})
    m = (model or "").lower()
    corpus_info = (params.get("corpora", {}).get(corpus, {}) or {}).get("description", "")
    evidence = collect_evidence(corpus, model, base_output, corpus_info=corpus_info)
    carregar_precos(params)
    _USO_CTX.update({"corpus": corpus, "modelo_topico": m, "fase": "site"})
    messages = build_viz_audit_messages(evidence, m)
    # Slugs sem CSV nesta rodada recebem o placeholder em CÓDIGO. Deixar o modelo
    # decidir isso era estocástico: com a mesma evidência, `similaridade_topicos`
    # abortou corretamente em 5 runs e produziu prosa especulativa em 3.
    com_dados, sem_dados = evidencia_por_slug(evidence, m)
    expected = [s for s, _, _ in com_dados]

    cfgs = painel_de_leituras(advisor_cfg)
    leituras, respostas, falhas, faltando = [], [], [], []
    for cfg in cfgs:
        try:
            sections, missing, response, cfg_used, cfg_retry = _uma_leitura(
                messages, cfg, expected, retry_on_missing)
        except Exception as exc:   # painelista fora do ar não derruba o outro
            falhas.append({"provider": cfg.get("provider"),
                           "model": cfg.get("model"), "erro": repr(exc)})
            continue
        for slug in sem_dados:
            sections.setdefault(slug, PLACEHOLDER_SEM_DADOS)
        leituras.append((cfg_used, sections))
        respostas.append(f"# {cfg_used.get('provider')}/{cfg_used.get('model')}"
                         f"\n\n{response}" if len(cfgs) > 1 else response)
        faltando.append((cfg_used, cfg_retry, missing))
    if not leituras:
        raise AdvisorConfigError(f"nenhum provider respondeu na fase site: {falhas}")

    # Faltante = slug que NENHUM painelista entregou. Com a união, um slug que
    # o provider A respondeu e o B não entrava em `viz_missing` mesmo com o
    # <slug>.md publicado com uma leitura íntegra — o ledger contradizia o disco
    # (o mesmo arquivo aparecia em `viz_written` E em `viz_missing`).
    entregues = set()
    for _cfg, secs in leituras:
        entregues |= set(secs)
    missing = [s for s in expected if s not in entregues]
    # As lacunas POR painelista continuam registradas: servem para saber que um
    # slug saiu com uma leitura só, sem marcar o arquivo como ausente.
    lacunas = {f"{u.get('provider')}/{u.get('model')}": faltam
               for u, _r, faltam in faltando if faltam}
    written = persist_viz_panel(evidence["run_dir"], corpus, model,
                                leituras, "\n\n".join(respostas))
    ledger_path = os.path.join(base_output, "llm_advisor_state.json")
    cfg_used, cfg_retry, _ = faltando[0]
    status = "audited" if not missing else "partial"
    if len(cfgs) > 1 and falhas:
        status = "partial_panel"
    entrada = {
        "round": resolve_round(ledger_path), "phase": "site",
        "status": status,
        "evidence_source": os.path.basename(str(evidence["run_dir"])),
        "provider": cfg_used.get("provider"), "model": cfg_used.get("model"),
        "viz_written": [os.path.basename(p) for p in written],
        "viz_missing": missing,
    }
    if len(leituras) > 1 or falhas:
        # Quem respondeu de fato, na ordem publicada — sem isto o ledger atribui
        # o arquivo inteiro (duas leituras) ao primeiro provider.
        entrada["painel"] = [{"provider": c.get("provider"), "model": c.get("model")}
                             for c, _s in leituras]
    if falhas:
        entrada["painel_falhas"] = falhas
    if lacunas:
        entrada["painel_lacunas"] = lacunas
    # Retry servido por outro provider que a 1ª chamada: registrar os dois, senão
    # o ledger atribui o arquivo inteiro a quem escreveu só parte dele.
    if (cfg_retry.get("provider"), cfg_retry.get("model")) != (
            cfg_used.get("provider"), cfg_used.get("model")):
        entrada["provider_retry"] = cfg_retry.get("provider")
        entrada["model_retry"] = cfg_retry.get("model")
    append_ledger(ledger_path, corpus, model, entrada)
    return {"run_dir": evidence["run_dir"], "written": written,
            "missing": missing, "painel": [c for c, _s in leituras],
            "falhas": falhas}
