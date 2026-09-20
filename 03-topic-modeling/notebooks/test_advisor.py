import glob
import json
import os
import re

import pytest

import _advisor


@pytest.fixture(autouse=True)
def _log_de_uso_isolado(tmp_path, monkeypatch):
    """Nenhum teste pode escrever no log de custo real.

    `_call_one` registra uso no ponto da chamada, entao qualquer teste que o
    exercite (mesmo com client mockado) gravava chamadas fantasma no JSONL do
    projeto e inflava o relatorio de gastos.
    """
    monkeypatch.setenv("ADVISOR_USAGE_LOG", str(tmp_path / "uso_teste.jsonl"))


# --- Evolução Task 1: prompts em YAML ----------------------------------------
def test_load_advisor_prompts_has_all_methods_and_phases():
    prompts = _advisor.load_advisor_prompts()
    assert set(prompts["methods"]) == {"stm"}
    for m in ("stm",):
        for key in ("system", "varredura", "final"):
            assert prompts["methods"][m][key].strip(), f"{m}.{key} vazio"
    assert "eval_metrics" in prompts["shared"]
    assert "output_contract" in prompts["shared"]
    # o contrato de saída menciona o bloco yaml e a chave proibida
    assert "yaml proposal" in prompts["shared"]["output_contract"]
    assert "stm_k" in prompts["shared"]["output_contract"]


def test_load_advisor_prompts_tolerates_partial_method_dirs(tmp_path):
    # metodo so com system+site (caso 'comparacao') nao pode quebrar o loader
    base = tmp_path / "advisor_prompts"
    (base / "comparacao").mkdir(parents=True)
    (base / "shared.yaml").write_text(
        "eval_metrics: x\noutput_contract: y\n", encoding="utf-8")
    (base / "comparacao" / "system.yaml").write_text("prompt: sys\n", encoding="utf-8")
    (base / "comparacao" / "site.yaml").write_text(
        "prompt: site {{VIZ_LIST}}\n", encoding="utf-8")
    data = _advisor.load_advisor_prompts(config_dir=str(tmp_path))
    assert data["methods"]["comparacao"]["system"] == "sys"
    assert "site" in data["methods"]["comparacao"]
    assert "varredura" not in data["methods"]["comparacao"]


# --- Task 1: validate_patch --------------------------------------------------
def test_validate_patch_accepts_bertopic_nmf_and_override_paths():
    patch = {
        "bertopic.hdbscan.min_cluster_size": 30,
        "corpora.tweets_bre2022.bertopic_overrides.reduce_outliers.threshold": 0.05,
        "evaluation.run_threshold_sweep": True,
        "nmf.k_range": [5, 30],
        "evaluation.nmf_kappa_grid": [0.5, 1.0, 2.0, 3.0, 4.0],
    }
    valid, invalid = _advisor.validate_patch(patch)
    assert invalid == []
    assert valid == patch


def test_validate_patch_rejects_unknown_and_forbidden_keys():
    patch = {
        "bertopic.hdbscan.min_cluster_size": 30,     # válida
        "evaluation.param_sweep_macro_k": [4, 5],    # proibida (inerte)
        "bertopic.foo.bar": 1,                        # desconhecida
    }
    valid, invalid = _advisor.validate_patch(patch)
    assert "bertopic.hdbscan.min_cluster_size" in valid
    assert set(invalid) == {"evaluation.param_sweep_macro_k", "bertopic.foo.bar"}
    assert "evaluation.param_sweep_macro_k" not in valid


def test_validate_patch_accepts_stm_grid_and_per_corpus_pins():
    patch = {
        "stm.k_range": [5, 30],
        "stm.max_em_its": 400,
        "evaluation.stm_sigma_prior_grid": [0, 0.3, 0.5, 0.7],
        "evaluation.stm_gamma_prior_grid": ["Pooled", "L1"],
        "corpora.youtube_doc.stm_best_k": 22,
        "corpora.folha.stm_sigma_prior": 0.5,
        "corpora.folha.stm_gamma_prior": "Pooled",
    }
    valid, invalid = _advisor.validate_patch(patch)
    assert invalid == []
    assert valid == patch


def test_validate_patch_rejects_stm_machine_config_paths():
    # rscript_path/timeout_sec são configuração de máquina, não hiperparâmetro
    # científico — nunca devem entrar no allowlist (ver Global Constraints).
    patch = {"stm.rscript_path": "C:/R/bin/Rscript.exe", "stm.timeout_sec": 60}
    valid, invalid = _advisor.validate_patch(patch)
    assert valid == {}
    assert set(invalid) == {"stm.rscript_path", "stm.timeout_sec"}


def test_validate_patch_rejects_wrong_typed_values():
    # nome permitido mas VALOR do tipo errado -> vai p/ invalid_keys.
    # param_sweep_min_samples tem de ser LISTA (vimos `: true`, que é bool).
    valid, invalid = _advisor.validate_patch({"evaluation.param_sweep_min_samples": True})
    assert valid == {}
    assert "evaluation.param_sweep_min_samples" in invalid
    # a mesma chave com uma LISTA é aceita
    valid, invalid = _advisor.validate_patch({"evaluation.param_sweep_min_samples": [3, 5, 10]})
    assert invalid == []
    assert valid == {"evaluation.param_sweep_min_samples": [3, 5, 10]}
    # threshold fora da faixa 0-1
    valid, invalid = _advisor.validate_patch({"bertopic.reduce_outliers.threshold": 1.5})
    assert "bertopic.reduce_outliers.threshold" in invalid
    # int esperado, string recebida
    valid, invalid = _advisor.validate_patch({"bertopic.hdbscan.min_cluster_size": "x"})
    assert "bertopic.hdbscan.min_cluster_size" in invalid


def test_validate_patch_type_checks_per_corpus_overrides_and_pins():
    # override per-corpus espelha o spec de bertopic.<suffix>; pin STM idem.
    valid, invalid = _advisor.validate_patch({
        "corpora.tweets_bre2022.bertopic_overrides.reduce_outliers.threshold": 2.0,
        "corpora.folha.stm_gamma_prior": "Nope",
    })
    assert valid == {}
    assert set(invalid) == {
        "corpora.tweets_bre2022.bertopic_overrides.reduce_outliers.threshold",
        "corpora.folha.stm_gamma_prior",
    }


# --- Task 2: parse_proposal --------------------------------------------------
_GOOD_RESPONSE = """Aqui vai minha análise.

```yaml proposal
round: 2
target_layer: outliers
params_patch:
  corpora.tweets_bre2022.bertopic_overrides.reduce_outliers.threshold: 0.05
  evaluation.param_sweep_macro_k: [4, 5]
sweep_grid:
  run_threshold_sweep: true
rationale_short: "threshold baixo captura reatribuições corretas"
```

## Pré-análise
O corpus mostra outliers altos; recomendo baixar o threshold.
"""


def test_parse_proposal_extracts_block_report_and_flags_invalid():
    proposal, invalid, report = _advisor.parse_proposal(_GOOD_RESPONSE)
    assert proposal["round"] == 2
    assert proposal["target_layer"] == "outliers"
    # forbidden key filtered out of the applied patch, but reported
    assert "evaluation.param_sweep_macro_k" in invalid
    assert list(proposal["params_patch"].keys()) == [
        "corpora.tweets_bre2022.bertopic_overrides.reduce_outliers.threshold"
    ]
    assert "Pré-análise" in report


def test_parse_proposal_captures_prose_before_the_block():
    # o modelo às vezes põe a ## Pré-análise ANTES do bloco yaml — o parser deve
    # capturar a prosa de fora do bloco (antes E depois), não só depois.
    resp = ("## Pré-análise\nAnálise que veio ANTES do bloco.\n\n"
            "```yaml proposal\nround: 1\ntarget_layer: lda_priors\n"
            "params_patch:\n  lda.k_range: [5, 30]\n```\n")
    proposal, invalid, report = _advisor.parse_proposal(resp)
    assert "Análise que veio ANTES do bloco" in report
    assert proposal["target_layer"] == "lda_priors"


def test_parse_proposal_raises_when_no_block():
    with pytest.raises(_advisor.ProposalFormatError):
        _advisor.parse_proposal("Sem bloco estruturado aqui.")


def test_parse_proposal_raises_when_missing_params_patch():
    bad = "```yaml proposal\nround: 1\ntarget_layer: structure\n```\n"
    with pytest.raises(_advisor.ProposalFormatError):
        _advisor.parse_proposal(bad)


# --- Task 3: ledger e detecção de rodada -------------------------------------
def test_resolve_round_empty_is_one(tmp_path):
    led = os.path.join(tmp_path, "llm_advisor_state.json")
    assert _advisor.resolve_round(led) == 1


def test_resolve_round_increments_and_respects_override(tmp_path):
    led = os.path.join(tmp_path, "llm_advisor_state.json")
    _advisor.append_ledger(led, "tweets_bre2022", "bertopic", {"round": 1, "status": "applied"})
    assert _advisor.resolve_round(led) == 2
    assert _advisor.resolve_round(led, override=1) == 1


def test_append_ledger_persists_metadata(tmp_path):
    led = os.path.join(tmp_path, "llm_advisor_state.json")
    _advisor.append_ledger(led, "folha", "lda", {"round": 1, "status": "proposed"})
    data = _advisor.read_ledger(led)
    assert data["corpus"] == "folha"
    assert data["model"] == "lda"
    assert data["rounds"][0]["round"] == 1


# --- Task 4: coleta de evidências --------------------------------------------
_REAL_BASE = os.path.join("..", "data", "output", "tweets_bre2022", "bertopic")


def test_collect_evidence_reads_recent_run_csvs():
    if not glob.glob(os.path.join(_REAL_BASE, "tweets_bre2022_*")):
        pytest.skip("no real bertopic run available")
    ev = _advisor.collect_evidence("tweets_bre2022", "bertopic", _REAL_BASE)
    assert ev["corpus"] == "tweets_bre2022"
    assert ev["run_dir"]  # non-empty path
    # ao menos UM CSV de evidência entrou. NÃO exigir 'metrics': folha/bertopic
    # não gera *metrics.csv (topics_for_eval/exclusividade cobrem); só tweets/LDA têm.
    assert ev["csv_snippets"]
    # snippets are strings, truncated
    for text in ev["csv_snippets"].values():
        assert isinstance(text, str)


def test_collect_evidence_gathers_sweep_from_run_without_results(tmp_path):
    # Determinístico (sem dados reais). Trava a divergência 2: a evidência de
    # sweep vive num run SEM results.csv e ainda assim deve ser coletada.
    base = str(tmp_path)
    older = os.path.join(base, "c_20260101_000000")  # sweep, SEM results.csv
    newer = os.path.join(base, "c_20260202_000000")  # produção, COM results.csv
    os.makedirs(older)
    os.makedirs(newer)
    with open(os.path.join(older, "sweep_bertopic_grid.csv"), "w", encoding="utf-8") as f:
        f.write("C_v,Diversity,Exclus,FREX,Stability\n0.6,0.7,0.5,0.4,0.8\n")
    with open(os.path.join(older, "sweep_bertopic_grid_raw.csv"), "w", encoding="utf-8") as f:
        f.write("seed,C_v\n42,0.6\n")  # variante _raw deve ser IGNORADA
    with open(os.path.join(newer, "bertopic_results.csv"), "w", encoding="utf-8") as f:
        f.write("topic,label\n0,x\n")
    ev = _advisor.collect_evidence("c", "bertopic", base)
    assert "sweep_bertopic_grid.csv" in ev["csv_snippets"]      # entrou, mesmo sem results.csv
    assert "sweep_bertopic_grid_raw.csv" not in ev["csv_snippets"]  # _raw filtrado
    assert os.path.basename(ev["run_dir"]) == "c_20260202_000000"   # primário = mais novo c/ results


def test_collect_evidence_includes_base_dir_grid_caches(tmp_path):
    # LDA/NMF: caches de grid (lda_alpha_eta_grid.csv / nmf_kappa_minprob_grid.csv)
    # e o *metrics.csv cacheado vivem na pasta-BASE (pai dos run dirs) — a
    # evidência da camada de hiperparâmetros vem de lá, não dos runs.
    base = str(tmp_path)
    run = os.path.join(base, "c_20260101_000000")
    os.makedirs(run)
    with open(os.path.join(run, "nmf_topics_for_eval.csv"), "w", encoding="utf-8") as f:
        f.write("topic_id,keywords\n0,eleicao lula\n")
    with open(os.path.join(base, "nmf_kappa_minprob_grid.csv"), "w", encoding="utf-8") as f:
        f.write("kappa,minimum_probability,cv\n1.0,0.01,0.56\n")
    with open(os.path.join(base, "nmf_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("model,n_topics,best_k_cv\nnmf,8,0.559\n")
    ev = _advisor.collect_evidence("c", "nmf", base)
    assert "nmf_kappa_minprob_grid.csv" in ev["csv_snippets"]  # cache da base entrou
    assert "nmf_metrics.csv" in ev["csv_snippets"]
    assert "nmf_topics_for_eval.csv" in ev["csv_snippets"]     # run dir continua coberto


def test_collect_evidence_backfills_topic_ranking_for_lda(tmp_path):
    # Item 2 (híbrido): sem *_exclusividade_ranking.csv, o advisor deriva
    # n_docs por tópico do *_results.csv (só p/ lda/nmf/stm).
    base = str(tmp_path)
    run = os.path.join(base, "c_20260101_000000")
    os.makedirs(run)
    with open(os.path.join(run, "lda_results.csv"), "w", encoding="utf-8") as f:
        f.write("post_id,topic_id,topic_name,granularity\n"
                "p0,0,Economia,unit\np1,0,Economia,unit\np2,1,Futebol,unit\n"
                "p3,0,MacroX,macro\n")  # linha macro deve ser IGNORADA
    ev = _advisor.collect_evidence("c", "lda", base)
    assert "derived_topic_ranking.csv" in ev["csv_snippets"]
    table = ev["csv_snippets"]["derived_topic_ranking.csv"]
    assert "n_docs" in table
    # tópico 0 (Economia) = 2 docs unit; macro não conta
    assert "Economia" in table and "Futebol" in table


def test_collect_evidence_no_backfill_when_ranking_exists(tmp_path):
    # se o pipeline já emite o ranking canônico, o backfill NÃO roda.
    base = str(tmp_path)
    run = os.path.join(base, "c_20260101_000000")
    os.makedirs(run)
    with open(os.path.join(run, "nmf_results.csv"), "w", encoding="utf-8") as f:
        f.write("post_id,topic_id,topic_name\np0,0,Economia\n")
    with open(os.path.join(run, "nmf_exclusividade_ranking.csv"), "w", encoding="utf-8") as f:
        f.write("topic_id,topic_name,exclusividade,n_docs\n0,Economia,0.8,1\n")
    ev = _advisor.collect_evidence("c", "nmf", base)
    assert "nmf_exclusividade_ranking.csv" in ev["csv_snippets"]  # canônico coletado
    assert "derived_topic_ranking.csv" not in ev["csv_snippets"]  # backfill NÃO roda


def test_collect_evidence_no_backfill_for_bertopic(tmp_path):
    # bertopic já emite o ranking próprio; o backfill é só p/ lda/nmf/stm.
    base = str(tmp_path)
    run = os.path.join(base, "c_20260101_000000")
    os.makedirs(run)
    with open(os.path.join(run, "bertopic_results.csv"), "w", encoding="utf-8") as f:
        f.write("post_id,topic_id,topic_name\np0,0,Economia\n")
    ev = _advisor.collect_evidence("c", "bertopic", base)
    assert "derived_topic_ranking.csv" not in ev["csv_snippets"]


def test_collect_evidence_includes_stm_grid_k_diagnostics(tmp_path):
    # stm_grid_k_diagnostics.csv (heldout likelihood/semantic coherence por K)
    # não casa com *_grid.csv (sufixo _diagnostics.csv) — precisa de glob próprio.
    base = str(tmp_path)
    with open(os.path.join(base, "stm_grid_k_diagnostics.csv"), "w", encoding="utf-8") as f:
        f.write("k,cv,heldout_likelihood\n25,0.677,-7.1\n")
    with open(os.path.join(base, "stm_sigma_gamma_grid.csv"), "w", encoding="utf-8") as f:
        f.write("sigma_prior,gamma_prior,cv\n0.7,L1,0.6799\n")
    ev = _advisor.collect_evidence("c", "stm", base)
    assert "stm_grid_k_diagnostics.csv" in ev["csv_snippets"]
    assert "stm_sigma_gamma_grid.csv" in ev["csv_snippets"]


# --- Correção 1: nmf_beta.csv/stm_beta.csv (matriz tópico x vocabulário) -----
def _wide_beta_csv(tmp_path, name, n_topics=5, n_words=200, with_topic_id=True):
    """Beta/phi sintético: colunas com pesos DECRESCENTES (word_0 sempre a maior),
    p/ o teste poder afirmar QUAIS colunas sobrevivem ao truncamento."""
    cols = [f"word_{i}" for i in range(n_words)]
    header = (["topic_id"] if with_topic_id else []) + cols
    linhas = [",".join(header)]
    for t in range(n_topics):
        # peso da coluna i decai com i -> soma total por coluna também decai com i
        vals = [str(round((n_words - i) * 0.01 + t, 4)) for i in range(n_words)]
        row = ([str(t)] if with_topic_id else []) + vals
        linhas.append(",".join(row))
    path = tmp_path / name
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(path)


def test_evidence_globs_match_nmf_and_stm_beta():
    import fnmatch
    for name in ("nmf_beta.csv", "stm_beta.csv"):
        assert any(fnmatch.fnmatch(name, pat) for pat in _advisor._EVIDENCE_GLOBS), name


def test_truncate_wide_csv_keeps_topic_id_and_top_columns_by_weight(tmp_path):
    path = _wide_beta_csv(tmp_path, "nmf_beta.csv", n_topics=3, n_words=200)
    out = _advisor._truncate_wide_csv(path, max_cols=10)
    header = out.splitlines()[0].split(",")
    assert header[0] == "topic_id"
    # as top-10 colunas por peso total são as PRIMEIRAS (word_0..word_9) —
    # construídas com peso decrescente
    assert header[1:] == [f"word_{i}" for i in range(10)]
    # linha por tópico preservada (3 tópicos + header)
    assert len(out.strip().splitlines()) == 4


def test_truncate_wide_csv_synthesizes_topic_id_when_absent(tmp_path):
    # stm_beta.csv não tem coluna topic_id (1a coluna já é uma palavra) — o
    # índice de linha vira o topic_id sintetizado.
    path = _wide_beta_csv(tmp_path, "stm_beta.csv", n_topics=4, n_words=150,
                          with_topic_id=False)
    out = _advisor._truncate_wide_csv(path, max_cols=8)
    header = out.splitlines()[0].split(",")
    assert header[0] == "topic_id"
    assert header[1:] == [f"word_{i}" for i in range(8)]
    linhas = out.strip().splitlines()
    assert len(linhas) == 5  # header + 4 tópicos
    assert linhas[1].startswith("0,")  # topic_id sintetizado começa em 0


def test_collect_evidence_includes_beta_csv_truncated(tmp_path):
    import pathlib
    base = str(tmp_path)
    run = pathlib.Path(base) / "c_20260101_000000"
    run.mkdir()
    _wide_beta_csv(run, "nmf_beta.csv", n_topics=5, n_words=500)
    ev = _advisor.collect_evidence("c", "nmf", base)
    assert "nmf_beta.csv" in ev["csv_snippets"]
    snippet = ev["csv_snippets"]["nmf_beta.csv"]
    # tamanho contido: bem menor que o CSV largo original (500 colunas -> poucas)
    n_cols_snippet = len(snippet.splitlines()[0].split(","))
    assert n_cols_snippet < 50


# --- Evolução Task 2: build_advisor_messages + select_phase ------------------
def _ev(corpus="youtube_doc", model="stm"):
    return {"corpus": corpus, "model": model, "run_dir": "run_x",
            "csv_snippets": {f"{model}_metrics.csv": "col\n1\n"}, "ledger_rounds": 0}


def test_select_phase_varredura_then_final():
    assert _advisor.select_phase(1, rounds=3) == "varredura"
    assert _advisor.select_phase(2, rounds=3) == "varredura"
    assert _advisor.select_phase(3, rounds=3) == "final"
    assert _advisor.select_phase(1, rounds=3, override="final") == "final"


def test_build_messages_first_round_has_system_and_user():
    msgs = _advisor.build_advisor_messages(_ev("youtube_doc", "stm"), round_n=1, model_type="stm")
    assert [m["role"] for m in msgs] == ["system", "user"]
    sys_low = msgs[0]["content"].lower()
    # system é específico do método (LDA) e cita as métricas de avaliação
    assert "stm" in sys_low and "semantic coherence" in sys_low and "sigma.prior" in sys_low
    assert "c_v" in sys_low and "diversity" in sys_low and "stability" not in sys_low
    user = msgs[1]["content"]
    # varredura injeta allowlist exato do LDA + evidência + contrato de saída
    assert "evaluation.stm_sigma_prior_grid" in user and "stm.k_range" in user
    assert "lda.k_range" not in user and "nmf.k_range" not in user and "bertopic" not in user
    assert "stm_metrics.csv" in user
    assert "yaml proposal" in user
    assert "stm_sigma_prior_grid" in user


def test_build_messages_final_phase_asks_two_part_report():
    msgs = _advisor.build_advisor_messages(_ev("youtube_doc", "stm"), round_n=3, model_type="stm")
    user = msgs[-1]["content"].lower()
    assert "retrospectiva" in user            # parte 1
    assert "pré-análise" in user or "pre-análise" in user  # parte 2
    assert "temas-ruído" in user or "temas-ruido" in user


def test_build_messages_with_history_appends_only_user():
    history = [{"role": "system", "content": "sys"},
               {"role": "user", "content": "u1"},
               {"role": "assistant", "content": "a1"}]
    msgs = _advisor.build_advisor_messages(_ev("youtube_doc", "stm"), round_n=2,
                                           model_type="stm", history=history)
    assert msgs[:3] == history            # histórico preservado
    assert msgs[-1]["role"] == "user"     # só o novo turno user é anexado
    assert len(msgs) == 4


def test_build_messages_stm_system_has_calibration_order():
    msgs = _advisor.build_advisor_messages(_ev("youtube_doc", "stm"),
                                           round_n=1, model_type="stm")
    sys_low = msgs[0]["content"].lower()
    assert "semantic coherence" in sys_low and "sigma.prior" in sys_low
    assert "outlier" in sys_low  # o prompt esclarece que STM nao possui topico de outlier


def _valid_value_for(path):
    """Um valor que satisfaz o spec de tipo/faixa da chave (p/ testes de allowlist)."""
    nk = _advisor._normalize_spec_key(path)
    if nk.endswith(".k_range") or nk.startswith("evaluation.") and (
        nk.endswith("_grid") or nk.endswith("_grids")
        or nk.startswith("evaluation.param_sweep_") or nk.endswith("_sweep_strategies")
    ):
        return [1, 2]
    if nk in ("evaluation.run_param_sweep", "evaluation.run_outlier_sweep",
              "evaluation.run_threshold_sweep"):
        return True
    if nk == "bertopic.hdbscan.cluster_selection_method":
        return "eom"
    if nk == "bertopic.reduce_outliers.strategy":
        return "off"
    if nk in ("bertopic.reduce_outliers.threshold", "stm_sigma_prior"):
        return 0.5
    if nk == "stm_gamma_prior":
        return "L1"
    return 2  # int / no_above (sem spec) — 2 é válido em ambos


def test_allowlist_for_model_stm_includes_per_corpus_pins():
    paths = _advisor._allowlist_for_model("stm", "youtube_doc")
    assert "evaluation.stm_sigma_prior_grid" in paths
    assert "corpora.youtube_doc.stm_best_k" in paths
    # todo caminho proposto passa pelo validador (consistência allowlist↔prompt)
    valid, invalid = _advisor.validate_patch({p: _valid_value_for(p) for p in paths})
    assert invalid == []


def test_build_messages_injects_corpus_info():
    # o corpus_info (tipo do corpus) evita o modelo alucinar "tweets" em notícias.
    fake_prompts = {
        "shared": {"output_contract": "OC", "eval_metrics": "EM"},
        "methods": {"lda": {
            "system": "sys {{EVAL_METRICS}}",
            "varredura": "corpus={{CORPUS}} info={{CORPUS_INFO}} round={{ROUND}}",
            "final": "f"}},
    }
    ev = _ev("folha", "lda")
    ev["corpus_info"] = "notícias jornalísticas, textos longos (PT-BR)"
    msgs = _advisor.build_advisor_messages(ev, round_n=1, model_type="lda",
                                           prompts=fake_prompts)
    user = msgs[-1]["content"]
    assert "info=notícias jornalísticas, textos longos (PT-BR)" in user
    assert "{{CORPUS_INFO}}" not in user


def test_fill_resolves_round_nested_in_output_contract():
    # o prompt-engineer pode escrever `round: {{ROUND}}` DENTRO do output_contract;
    # a reordenação do _fill garante que o token aninhado resolva.
    fake_prompts = {
        "shared": {"output_contract": "contrato round: {{ROUND}}", "eval_metrics": "EM"},
        "methods": {"lda": {"system": "s", "varredura": "{{OUTPUT_CONTRACT}}", "final": "f"}},
    }
    msgs = _advisor.build_advisor_messages(_ev("folha", "lda"), round_n=7,
                                           model_type="lda", phase="varredura",
                                           prompts=fake_prompts)
    user = msgs[-1]["content"]
    assert "round: 7" in user
    assert "{{ROUND}}" not in user


# --- Task 6: chamada à API (OpenAI / Ollama) ---------------------------------
def test_call_advisor_raises_when_disabled():
    with pytest.raises(_advisor.AdvisorConfigError):
        _advisor.call_advisor([{"role": "user", "content": "x"}], {"enabled": False})


def test_call_advisor_openai_raises_when_key_env_missing(monkeypatch):
    monkeypatch.delenv("ADVISOR_TEST_KEY", raising=False)
    cfg = {"enabled": True, "provider": "openai", "model": "gpt-4o",
           "api_key_env": "ADVISOR_TEST_KEY"}
    with pytest.raises(_advisor.AdvisorConfigError):
        _advisor.call_advisor([{"role": "user", "content": "x"}], cfg)


def test_call_advisor_openai_happy_path(monkeypatch):
    monkeypatch.setenv("ADVISOR_TEST_KEY", "sk-test")
    cfg = {"enabled": True, "provider": "openai", "model": "gpt-4o",
           "api_key_env": "ADVISOR_TEST_KEY", "max_output_tokens": 100, "temperature": 0.2}
    captured = {}

    class _FakeMsg:
        content = "resposta do modelo"

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return type("R", (), {"choices": [_FakeChoice()]})()

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    import sys, types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "prompt"}]
    out, cfg_used = _advisor.call_advisor(msgs, cfg)
    assert out == "resposta do modelo"
    assert captured["api_key"] == "sk-test"
    assert captured["create"]["model"] == "gpt-4o"
    assert captured["create"]["messages"] == msgs   # lista repassada intacta
    # cfg efetivamente usada = a própria cfg primária (sem fallback envolvido)
    assert cfg_used is cfg


def test_call_advisor_ollama_needs_no_key(monkeypatch):
    # Ollama local: SEM chave de API — usa base_url do endpoint OpenAI-compat.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = {"enabled": True, "provider": "ollama", "model": "gemma2:2b-instruct-q4_K_M",
           "base_url": "http://localhost:11434/v1", "temperature": 0.2}
    captured = {}

    class _FakeMsg:
        content = "resposta local"

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return type("R", (), {"choices": [_FakeChoice()]})()

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    import sys, types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    out, cfg_used = _advisor.call_advisor([{"role": "user", "content": "prompt"}], cfg)
    assert out == "resposta local"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["create"]["model"] == "gemma2:2b-instruct-q4_K_M"
    assert cfg_used is cfg


def test_call_advisor_ollama_uses_api_key_env_when_set(monkeypatch):
    # Ollama remoto/cloud autenticado: lê a chave de api_key_env (ex.: OLLAMA_API_KEY).
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-cloud-key")
    cfg = {"enabled": True, "provider": "ollama", "model": "qwen3:8b",
           "base_url": "https://ollama.com/v1", "api_key_env": "OLLAMA_API_KEY"}
    captured = {}

    class _FakeMsg:
        content = "ok"

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeCompletions:
        def create(self, **kwargs):
            return type("R", (), {"choices": [_FakeChoice()]})()

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    import sys, types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    _advisor.call_advisor([{"role": "user", "content": "prompt"}], cfg)
    assert captured["api_key"] == "ollama-cloud-key"   # chave real, não a dummy
    assert captured["base_url"] == "https://ollama.com/v1"


def test_call_advisor_falls_back_to_ollama_when_primary_raises(monkeypatch):
    # primary (openai/gpt) COM chave, mas a chamada estoura (erro de rede/rate
    # limit simulado) -> fallback automático p/ ollama remoto (qwen3:8b).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = {
        "enabled": True, "provider": "openai", "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY", "base_url": None,
        "temperature": 0.2, "max_output_tokens": 4000,
        "fallback": {"provider": "ollama", "model": "qwen3:8b",
                     "base_url": "http://remoto:11434/v1"},
    }
    captured = {}

    class _FakeMsg:
        content = "resposta do fallback"

    class _FakeChoice:
        message = _FakeMsg()

    class _FakeCompletions:
        def create(self, **kwargs):
            if kwargs["model"] == "gpt-4o":
                raise RuntimeError("simulated network error")  # primary falha
            captured["create"] = kwargs
            return type("R", (), {"choices": [_FakeChoice()]})()

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["base_url"] = base_url
            self.chat = type("C", (), {"completions": _FakeCompletions()})()

    import sys, types
    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    out, cfg_used = _advisor.call_advisor([{"role": "user", "content": "prompt"}], cfg)
    assert out == "resposta do fallback"
    # o client foi reconstruído com o base_url do fallback (Ollama remoto)
    assert captured["base_url"] == "http://remoto:11434/v1"
    assert captured["create"]["model"] == "qwen3:8b"
    # temperature herdada do bloco pai (fallback não a definiu)
    assert captured["create"]["temperature"] == 0.2
    # Correção 2 (proveniência): a cfg efetivamente usada é a do FALLBACK, não a
    # do primary — quem consome o retorno (rodapé, ledger) tem de atribuir a
    # resposta ao modelo que respondeu de verdade.
    assert cfg_used["provider"] == "ollama"
    assert cfg_used["model"] == "qwen3:8b"


# --- Task 7: persistência da rodada ------------------------------------------
def test_persist_round_writes_files_and_ledger(tmp_path):
    base = str(tmp_path)
    proposal = {"round": 1, "target_layer": "structure",
                "params_patch": {"bertopic.hdbscan.min_cluster_size": 30}}
    paths = _advisor.persist_round(
        base, "tweets_bre2022", "bertopic", 1,
        evidence={"run_dir": "run_x"}, proposal=proposal, invalid_keys=[],
        report_md="## Pré-análise\nok", status="proposed",
        cycle_round=1, phase="varredura",
    )
    assert os.path.exists(paths["proposal_file"])
    assert os.path.exists(paths["report_file"])
    led = _advisor.read_ledger(os.path.join(base, "llm_advisor_state.json"))
    assert led["rounds"][0]["round"] == 1
    assert led["rounds"][0]["cycle_round"] == 1
    assert led["rounds"][0]["phase"] == "varredura"
    assert led["rounds"][0]["status"] == "proposed"
    assert led["rounds"][0]["evidence_source"] == "run_x"


def test_persist_round_records_effective_provider_and_model_when_given(tmp_path):
    # Correção 2: persist_round aceita a cfg EFETIVAMENTE usada (pode ser o
    # fallback) e grava provider/model na entrada do ledger. Retrocompatível:
    # sem advisor_cfg (teste acima), a entrada simplesmente não ganha as chaves.
    base = str(tmp_path)
    proposal = {"round": 1, "target_layer": "structure", "params_patch": {}}
    _advisor.persist_round(
        base, "youtube_doc", "stm", 1,
        evidence={"run_dir": "run_x"}, proposal=proposal, invalid_keys=[],
        report_md="ok", status="proposed", cycle_round=1, phase="varredura",
        advisor_cfg={"provider": "ollama", "model": "qwen3:8b"},
    )
    led = _advisor.read_ledger(os.path.join(base, "llm_advisor_state.json"))
    assert led["rounds"][0]["provider"] == "ollama"
    assert led["rounds"][0]["model"] == "qwen3:8b"


def test_run_advisor_round_ledger_records_effective_provider_on_fallback(tmp_path, monkeypatch):
    # mesma correção, ponta a ponta pelo orquestrador run_advisor_round: o
    # ledger tem de citar o provider/model que RESPONDEU, não o primary
    # configurado (mesmo cenário de bug do run_viz_audit).
    base = str(tmp_path)
    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "run_dir": "run_x",
        "csv_snippets": {}, "ledger_rounds": 0})
    fallback_cfg = {"provider": "ollama", "model": "qwen3:8b"}
    monkeypatch.setattr(_advisor, "call_advisor",
                        lambda messages, cfg: (_GOOD_RESPONSE, fallback_cfg))
    _advisor.run_advisor_round(
        "youtube_doc", "stm", base,
        params={"advisor": {"enabled": True, "provider": "openai", "model": "gpt-4o",
                            "api_key_env": "X", "rounds": 3, "fallback": fallback_cfg}})
    led = _advisor.read_ledger(os.path.join(base, "llm_advisor_state.json"))
    assert led["rounds"][-1]["provider"] == "ollama"
    assert led["rounds"][-1]["model"] == "qwen3:8b"


# --- Evolução Task 4: memória de chat ----------------------------------------
def test_read_chat_absent_is_empty(tmp_path):
    assert _advisor.read_chat(os.path.join(tmp_path, "advisor_chat.json")) == []


def test_save_and_read_chat_roundtrip(tmp_path):
    chat = os.path.join(tmp_path, "advisor_chat.json")
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"}]
    _advisor.save_chat(chat, "youtube_doc", "stm", msgs)
    assert _advisor.read_chat(chat) == msgs


# --- Task 8: orquestrador (com memória de chat) ------------------------------
def test_run_advisor_round_happy_path(tmp_path, monkeypatch):
    base = str(tmp_path)

    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "run_dir": "run_x",
        "csv_snippets": {}, "ledger_rounds": 0})
    monkeypatch.setattr(_advisor, "call_advisor", lambda messages, cfg: (_GOOD_RESPONSE, cfg))

    result = _advisor.run_advisor_round(
        "youtube_doc", "stm", base,
        params={"advisor": {"enabled": True, "api_key_env": "X", "rounds": 3}})
    assert result["round"] == 1
    assert result["phase"] == "varredura"
    assert result["status"] == "proposed"
    assert "evaluation.param_sweep_macro_k" in result["invalid_keys"]
    assert os.path.exists(result["paths"]["proposal_file"])
    # memória de chat criada: system + user + assistant
    chat = _advisor.read_chat(os.path.join(base, "advisor_chat.json"))
    assert [m["role"] for m in chat] == ["system", "user", "assistant"]
    assert chat[-1]["content"] == _GOOD_RESPONSE


def test_run_advisor_round_second_round_replays_history(tmp_path, monkeypatch):
    base = str(tmp_path)
    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "run_dir": "run_x",
        "csv_snippets": {}, "ledger_rounds": 1})
    seen = {}

    def _fake_call(messages, cfg):
        seen["n"] = len(messages)
        return _GOOD_RESPONSE, cfg
    monkeypatch.setattr(_advisor, "call_advisor", _fake_call)
    # pré-popula uma conversa de rodada 1 (system+user+assistant)
    _advisor.save_chat(os.path.join(base, "advisor_chat.json"), "youtube_doc", "stm",
                       [{"role": "system", "content": "s"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"}])
    result = _advisor.run_advisor_round(
        "youtube_doc", "stm", base, round=2,
        params={"advisor": {"enabled": True, "api_key_env": "X", "rounds": 3}})
    assert result["round"] == 2
    # a rodada 2 reenviou o histórico (3) + o novo turno user (1) = 4
    assert seen["n"] == 4
    chat = _advisor.read_chat(os.path.join(base, "advisor_chat.json"))
    assert len(chat) == 5  # + assistant da rodada 2


def test_run_advisor_round_retry_then_succeed_keeps_chat_clean(tmp_path, monkeypatch):
    # 1ª resposta malformada, retry corrige. A memória de chat deve guardar SÓ o
    # turno limpo (system, user, assistant-bom) — nem a resposta ruim nem o
    # lembrete corretivo, senão o replay das próximas rodadas fica poluído.
    base = str(tmp_path)
    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "run_dir": "run_x",
        "csv_snippets": {}, "ledger_rounds": 0})
    calls = {"n": 0}

    def _fake_call(messages, cfg):
        calls["n"] += 1
        text = "lixo sem bloco" if calls["n"] == 1 else _GOOD_RESPONSE
        return text, cfg
    monkeypatch.setattr(_advisor, "call_advisor", _fake_call)

    result = _advisor.run_advisor_round(
        "youtube_doc", "stm", base,
        params={"advisor": {"enabled": True, "api_key_env": "X", "rounds": 3}})
    assert result["status"] == "proposed"
    assert calls["n"] == 2  # houve retry
    chat = _advisor.read_chat(os.path.join(base, "advisor_chat.json"))
    assert [m["role"] for m in chat] == ["system", "user", "assistant"]
    assert chat[-1]["content"] == _GOOD_RESPONSE
    blob = "".join(m["content"] for m in chat)
    assert "lixo sem bloco" not in blob                 # resposta ruim descartada
    assert "estritamente no formato" not in blob         # lembrete corretivo descartado


def test_run_advisor_round_needs_review_on_bad_format(tmp_path, monkeypatch):
    base = str(tmp_path)
    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "run_dir": "run_y",
        "csv_snippets": {}, "ledger_rounds": 0})
    monkeypatch.setattr(_advisor, "call_advisor", lambda messages, cfg: ("sem bloco válido", cfg))

    result = _advisor.run_advisor_round(
        "youtube_doc", "stm", base,
        params={"advisor": {"enabled": True, "api_key_env": "X", "rounds": 3}})
    assert result["status"] == "needs_review"
    assert os.path.exists(result["paths"]["report_file"])
    # needs_review NÃO cresce a memória de chat (nada de turnos malformados salvos)
    assert _advisor.read_chat(os.path.join(base, "advisor_chat.json")) == []


def test_run_advisor_round_final_resets_chat(tmp_path, monkeypatch):
    # fim de ciclo: uma rodada 'final' (relatório) fecha o ciclo e RESETA o chat,
    # p/ a próxima chamada recomeçar limpo na rodada 1/varredura.
    base = str(tmp_path)
    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "corpus_info": "", "run_dir": "run_x",
        "csv_snippets": {}, "ledger_rounds": 2})
    monkeypatch.setattr(_advisor, "call_advisor", lambda messages, cfg: (_GOOD_RESPONSE, cfg))
    # pré-popula 2 turnos assistant -> cycle_round=3 -> fase 'final' (rounds=3)
    _advisor.save_chat(os.path.join(base, "advisor_chat.json"), "youtube_doc", "stm",
                       [{"role": "system", "content": "s"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                        {"role": "assistant", "content": "a2"}])
    result = _advisor.run_advisor_round(
        "youtube_doc", "stm", base,
        params={"advisor": {"enabled": True, "api_key_env": "X", "rounds": 3}})
    assert result["phase"] == "final"
    assert result["status"] == "proposed"
    # chat resetado -> vazio para o próximo ciclo
    assert _advisor.read_chat(os.path.join(base, "advisor_chat.json")) == []


def test_run_advisor_round_cycle_round_from_chat(tmp_path, monkeypatch):
    # a fase vem da POSIÇÃO no ciclo (chat), não do round global do ledger.
    base = str(tmp_path)
    monkeypatch.setattr(_advisor, "collect_evidence", lambda *a, **k: {
        "corpus": "youtube_doc", "model": "stm", "corpus_info": "", "run_dir": "run_x",
        "csv_snippets": {}, "ledger_rounds": 0})
    seen = {}

    def _fake_call(messages, cfg):
        seen["round_token"] = messages[-1]["content"]
        return _GOOD_RESPONSE, cfg
    monkeypatch.setattr(_advisor, "call_advisor", _fake_call)
    # 2 turnos assistant no chat -> próxima é a rodada 3 do ciclo
    _advisor.save_chat(os.path.join(base, "advisor_chat.json"), "youtube_doc", "stm",
                       [{"role": "system", "content": "s"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "u2"},
                        {"role": "assistant", "content": "a2"}])
    result = _advisor.run_advisor_round(
        "youtube_doc", "stm", base,
        params={"advisor": {"enabled": True, "api_key_env": "X", "rounds": 3}})
    assert result["cycle_round"] == 3
    assert result["phase"] == "final"          # rounds=3 -> cycle 3 é final
    assert result["round"] == 1                # ledger vazio -> round global 1


def test_build_messages_unknown_model_raises(tmp_path):
    with pytest.raises(_advisor.AdvisorConfigError):
        _advisor.build_advisor_messages(_ev("folha", "lda"), round_n=1, model_type="gpt")


# --- Plano 2 / Task 2: fase site — registry de slugs e mensagens -------------
def test_viz_slugs_cover_all_models_and_are_wellformed():
    for m in ("lda", "nmf", "stm", "bertopic", "comparacao"):
        assert m in _advisor.VIZ_SLUGS and _advisor.VIZ_SLUGS[m]
        for slug, titulo in _advisor.VIZ_SLUGS[m]:
            assert re.fullmatch(r"[a-z0-9_]+", slug), slug
            assert titulo


def test_build_viz_audit_messages_lists_slugs_and_evidence():
    ev = {"corpus": "folha", "corpus_info": "noticias",
          "csv_snippets": {"lda_metrics.csv": "a,b\n1,2\n"}, "run_dir": "x"}
    prompts = {"shared": {"eval_metrics": "MET", "viz_audit_contract": "CONTRATO"},
               "methods": {"lda": {"system": "SYS {{EVAL_METRICS}}",
                                   "site": "AUDIT {{CORPUS}} {{VIZ_LIST}} {{EVIDENCE}} {{VIZ_AUDIT_CONTRACT}}"}}}
    msgs = _advisor.build_viz_audit_messages(ev, "lda", prompts=prompts)
    assert msgs[0]["role"] == "system" and "MET" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "folha" in user and "CONTRATO" in user and "lda_metrics.csv" in user
    # Só os slugs que o CSV desta rodada sustenta entram na lista pedida. Antes
    # TODOS entravam e o modelo tinha de adivinhar o pareamento — origem dos
    # textos errados publicados (ver VIZ_EVIDENCE em _advisor.py).
    assert "grid_k" in user and "docs_por_topico" in user
    assert "similaridade_topicos" not in user      # sem CSV em run nenhum
    assert "tsne_theta" not in user
    assert "k_grid_scores" in user                 # glossário de colunas junto


def test_build_viz_audit_messages_raises_without_site_phase():
    prompts = {"shared": {"eval_metrics": "M", "viz_audit_contract": "C"},
               "methods": {"lda": {"system": "S"}}}
    with pytest.raises(_advisor.AdvisorConfigError):
        _advisor.build_viz_audit_messages({"csv_snippets": {}}, "lda", prompts=prompts)


# --- Plano 2 / extra: evidencia derivada p/ cross-tab e serie temporal -------
def _corpus_csv(tmp_path, n_meses, categorias=("a", "b")):
    """corpus_limpo.csv minimo: post_id, data, category (n_meses meses seguidos)."""
    linhas = ["post_id,data,category"]
    for i in range(n_meses):
        ano, mes = 2020 + i // 12, i % 12 + 1
        for j, cat in enumerate(categorias):
            linhas.append(f"p{i}_{j},{ano}-{mes:02d}-15,{cat}")
    path = tmp_path / "corpus_limpo.csv"
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(path)


def _results_csv(tmp_path, corpus_csv, topicos=(0, 1)):
    """results.csv alinhado ao corpus: topic_id ciclando entre `topicos`."""
    import csv
    with open(corpus_csv, encoding="utf-8") as f:
        ids = [r["post_id"] for r in csv.DictReader(f)]
    linhas = ["post_id,topic_id,topic_name,granularity"]
    for i, pid in enumerate(ids):
        t = topicos[i % len(topicos)]
        linhas.append(f"{pid},{t},Tema {t},unit")
    path = tmp_path / "lda_results.csv"
    path.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return str(path)


def test_derive_topic_category_returns_wide_pct_table(tmp_path):
    corpus = _corpus_csv(tmp_path, n_meses=3, categorias=("esporte", "ciencia"))
    results = _results_csv(tmp_path, corpus)
    out = _advisor._derive_topic_category_from_results(results, corpus)
    linhas = out.strip().splitlines()
    assert linhas[0].startswith("category,n_docs")
    assert len(linhas) == 3                      # header + 2 categorias
    assert "esporte" in out and "ciencia" in out
    # percentuais por linha somam ~100
    corpo = [l.split(",") for l in linhas[1:]]
    for row in corpo:
        assert abs(sum(float(v) for v in row[2:]) - 100.0) < 0.5


def test_derive_topic_timeline_is_always_monthly(tmp_path):
    # A granularidade tem de bater com a das FIGURAS publicadas, que sao mensais.
    # Antes agregava por ano acima de 24 periodos: a folha (81 meses) caia nisso e
    # o LLM escrevia sobre "eventos especificos" sem nunca ter visto um mes.
    corpus = _corpus_csv(tmp_path, n_meses=3)
    curto = _advisor._derive_topic_timeline_from_results(_results_csv(tmp_path, corpus), corpus)
    assert curto.splitlines()[0].startswith("mes,")
    assert len(curto.strip().splitlines()) == 4          # header + 3 meses

    tmp2 = tmp_path / "longo"
    tmp2.mkdir()
    corpus_longo = _corpus_csv(tmp2, n_meses=36)
    longo = _advisor._derive_topic_timeline_from_results(
        _results_csv(tmp2, corpus_longo), corpus_longo)
    assert longo.splitlines()[0].startswith("mes,")
    assert len(longo.strip().splitlines()) == 37         # header + 36 meses, sem agregar


def test_collect_evidence_adds_derived_category_and_timeline(tmp_path, monkeypatch):
    base = tmp_path / "lda"
    run = base / "folha_20260101_000000"
    run.mkdir(parents=True)
    corpus = _corpus_csv(tmp_path, n_meses=2, categorias=("esporte",))
    results = _results_csv(tmp_path, corpus)
    (run / "lda_results.csv").write_text(
        open(results, encoding="utf-8").read(), encoding="utf-8")
    monkeypatch.setattr(_advisor, "_find_corpus_csv", lambda corpus_id: corpus)
    ev = _advisor.collect_evidence("folha", "lda", str(base))
    assert "derived_topic_category.csv" in ev["csv_snippets"]
    assert "derived_topic_timeline.csv" in ev["csv_snippets"]


def test_derive_stm_prevalence_keeps_significant_effects_only(tmp_path):
    p = tmp_path / "stm_prevalence_effects.csv"
    p.write_text(
        "topic_id,term,estimate,std_error,p_value\n"
        "0,(Intercept),0.9,0.1,0.001\n"          # intercepto: descartado
        "0,categoriaesporte,0.5,0.1,0.01\n"      # significativo
        "1,categoriaciencia,0.8,0.2,0.40\n"      # nao significativo
        "2,categoriasaude,-0.7,0.1,0.02\n",      # significativo (negativo)
        encoding="utf-8")
    out = _advisor._derive_stm_prevalence_summary(str(p))
    assert "(Intercept)" not in out
    assert "categoriaciencia" not in out
    linhas = out.strip().splitlines()
    assert len(linhas) == 3                       # header + 2 significativos
    assert linhas[1].startswith("2,")             # ordenado por |estimate| desc


@pytest.mark.skip(reason="BERTopic evidence fixture removed in STM-only repository")
def test_collect_evidence_picks_up_frex_macro_and_prevalence(tmp_path, monkeypatch):
    run = base / "folha_20260101_000000"
    run.mkdir(parents=True)
    (run / "stm_results.csv").write_text("post_id,topic_id,granularity\n1,0,unit\n",
                                         encoding="utf-8")
    (run / "stm_topics_frex.csv").write_text("topic_id,keywords_frex\n0,a b c\n", encoding="utf-8")
    (run / "bertopic_macro_temas.csv").write_text("topic_id,macro_id,n_docs\n0,1,10\n",
                                                  encoding="utf-8")
    (run / "bertopic_categoria_topico.csv").write_text("category,0\nesporte,5\n", encoding="utf-8")
    (run / "stm_prevalence_effects.csv").write_text(
        "topic_id,term,estimate,std_error,p_value\n0,categoriaesporte,0.5,0.1,0.01\n",
        encoding="utf-8")
    monkeypatch.setattr(_advisor, "_find_corpus_csv", lambda corpus_id: None)
    ev = _advisor.collect_evidence("youtube_doc", "stm", str(base))
    for nome in ("stm_topics_frex.csv", "bertopic_macro_temas.csv",
                 "bertopic_categoria_topico.csv",
                 "derived_stm_prevalence_significativos.csv"):
        assert nome in ev["csv_snippets"], nome


def test_collect_evidence_survives_missing_corpus_csv(tmp_path, monkeypatch):
    base = tmp_path / "stm"
    run = base / "youtube_doc_20260101_000000"
    run.mkdir(parents=True)
    (run / "stm_results.csv").write_text("post_id,topic_id,granularity\n1,0,unit\n",
                                         encoding="utf-8")
    monkeypatch.setattr(_advisor, "_find_corpus_csv", lambda corpus_id: None)
    ev = _advisor.collect_evidence("youtube_doc", "stm", str(base))
    assert "derived_topic_category.csv" not in ev["csv_snippets"]


# --- Plano 2 / Task 3: parser e persistencia da fase site --------------------
def test_parse_viz_audit_splits_sections_and_reports_missing():
    resp = ("intro ignorada\n## heatmap_phi\nanálise A\nlinha 2\n"
            "## grid_k\nanálise B\n## desconhecido\nignorar\n")
    sections, missing = _advisor.parse_viz_audit(resp, ["heatmap_phi", "grid_k", "tsne_theta"])
    assert sections["heatmap_phi"] == "análise A\nlinha 2"
    assert sections["grid_k"].startswith("análise B")
    assert "desconhecido" not in sections
    assert missing == ["tsne_theta"]


def test_run_viz_audit_persists_and_retries_missing(tmp_path, monkeypatch):
    base = tmp_path / "stm"
    run = base / "youtube_doc_20260722_000000"
    run.mkdir(parents=True)
    (run / "stm_results.csv").write_text("post_id,topic_id\n1,0\n", encoding="utf-8")
    (run / "stm_metrics.csv").write_text("k_grid_scores\n1\n", encoding="utf-8")
    (run / "stm_topics_frex.csv").write_text("topic_id,keywords_frex\n0,a b\n", encoding="utf-8")
    ev = _advisor.collect_evidence("youtube_doc", "stm", str(base))
    com, sem = _advisor.evidencia_por_slug(ev, "stm")
    slugs = [s for s, _, _ in com]
    assert len(slugs) >= 2 and sem, "fixture precisa de slugs com e sem evidencia"
    first = "\n".join(f"## {s}\nprosa {s}" for s in slugs[:-1])   # falta o último
    second = f"## {slugs[-1]}\nprosa retry"
    calls = []

    def fake_call(messages, cfg):
        calls.append(messages)
        return (first, cfg) if len(calls) == 1 else (second, cfg)

    monkeypatch.setattr(_advisor, "call_advisor", fake_call)
    params = {"advisor": {"provider": "openai", "model": "gpt-4o"},
              "corpora": {"youtube_doc": {"description": "noticias"}}}
    out = _advisor.run_viz_audit("youtube_doc", "stm", str(base), params=params)
    assert len(calls) == 2 and out["missing"] == []
    assert (run / "advisor_viz" / f"{slugs[-1]}.md").exists()
    # Slug sem CSV recebe o placeholder em codigo, sem depender do juizo do LLM
    sem_txt = (run / "advisor_viz" / f"{sem[0]}.md").read_text(encoding="utf-8")
    assert _advisor.PLACEHOLDER_SEM_DADOS in sem_txt
    led = _advisor.read_ledger(str(base / "llm_advisor_state.json"))
    assert led["rounds"][-1]["phase"] == "site"


def test_run_viz_audit_footer_and_ledger_use_effective_provider_on_fallback(tmp_path, monkeypatch):
    # Correção 2: se o PRIMARY falha e o FALLBACK responde, o rodapé publicado
    # e a entrada do ledger têm de atribuir a resposta ao FALLBACK (quem
    # respondeu de verdade), não ao provider/model configurados como primary.
    base = tmp_path / "stm"
    run = base / "youtube_doc_20260722_000000"
    run.mkdir(parents=True)
    (run / "stm_results.csv").write_text("post_id,topic_id\n1,0\n", encoding="utf-8")
    slugs = [s for s, _ in _advisor.VIZ_SLUGS["stm"]]
    resp = "\n".join(f"## {s}\nprosa {s}" for s in slugs)  # todas as seções, sem retry
    fallback_cfg = {"provider": "ollama", "model": "qwen3:8b"}

    def fake_call(messages, cfg):
        # simula o comportamento real de call_advisor: primary falhou por
        # baixo, fallback respondeu -> cfg_used devolvida é a do FALLBACK.
        return resp, fallback_cfg

    monkeypatch.setattr(_advisor, "call_advisor", fake_call)
    params = {"advisor": {"provider": "openai", "model": "gpt-4o",
                          "fallback": fallback_cfg},
              "corpora": {"youtube_doc": {"description": "noticias"}}}
    _advisor.run_viz_audit("youtube_doc", "stm", str(base), params=params)
    body = (run / "advisor_viz" / f"{slugs[0]}.md").read_text(encoding="utf-8")
    assert "ollama/qwen3:8b" in body
    assert "openai/gpt-4o" not in body
    led = _advisor.read_ledger(str(base / "llm_advisor_state.json"))
    assert led["rounds"][-1]["provider"] == "ollama"
    assert led["rounds"][-1]["model"] == "qwen3:8b"


def test_persist_viz_audit_writes_one_file_per_slug(tmp_path):
    run_dir = tmp_path / "youtube_doc_20260722_000000"
    run_dir.mkdir()
    cfg = {"provider": "openai", "model": "gpt-4o"}
    written = _advisor.persist_viz_audit(
        str(run_dir), "youtube_doc", "stm",
        {"heatmap_phi": "prosa X", "grid_k": "prosa Y"}, "RESPOSTA COMPLETA", cfg)
    assert sorted(os.path.basename(p) for p in written) == ["grid_k.md", "heatmap_phi.md"]
    body = (run_dir / "advisor_viz" / "heatmap_phi.md").read_text(encoding="utf-8")
    assert "prosa X" in body and "gpt-4o" in body
    # O rodape tem que carregar o aviso de IA: e o unico ponto onde o leitor do
    # site e avisado de que a pre-analise pode conter erro factual.
    assert "gerado por IA" in body and "Pode conter erros" in body
    assert (run_dir / "advisor_viz_report.md").read_text(encoding="utf-8") == "RESPOSTA COMPLETA"


# --- Fase site: robustez do parser, sanitizacao e nao-destruicao -------------
@pytest.mark.parametrize("cabecalho", [
    "## heatmap_phi",
    "### heatmap_phi",                 # H3 em vez de H2
    "## heatmap_phi — Heatmap φ",      # titulo residual
    "## 1. heatmap_phi",               # numerado
    "## **heatmap_phi**",              # enfase markdown
    "## Heatmap_phi",                  # maiuscula
    "## heatmap phi",                  # espaco no lugar do underscore
])
def test_parse_viz_audit_tolerates_header_variants(cabecalho):
    # Antes so '## slug' exato casava; qualquer desvio fazia a linha virar CORPO
    # da secao anterior (contaminando o arquivo anterior) E marcava o slug como
    # faltante, disparando um retry caro que podia repetir o mesmo desvio.
    resp = f"{cabecalho}\nanalise A\n## grid_k\nanalise B\n"
    sections, missing = _advisor.parse_viz_audit(resp, ["heatmap_phi", "grid_k"])
    assert sections["heatmap_phi"] == "analise A"
    assert sections["grid_k"] == "analise B"
    assert missing == []


def test_parse_viz_audit_survives_none_response():
    # O SDK devolve content=None quando o provider corta a geracao; antes isso
    # era AttributeError DEPOIS de a chamada ja ter sido paga.
    sections, missing = _advisor.parse_viz_audit(None, ["grid_k"])
    assert sections == {} and missing == ["grid_k"]


@pytest.mark.parametrize("bruto,proibido", [
    ("texto\n---\nmais texto", "\n---\n"),        # Pandoc leria como metadados YAML
    ("texto\n::: {.callout}\nx", "\n::: "),       # fecharia o callout do site
    ("antes <think>ruido</think> depois", "<think>"),
    ("prosa\n{{< include x.md >}}", "{{<"),       # Quarto tentaria resolver
])
def test_sanitize_viz_markdown_neutraliza_vetores_de_quebra(bruto, proibido):
    assert proibido not in _advisor.sanitize_viz_markdown(bruto)


def test_sanitize_viz_markdown_fecha_cerca_impar():
    out = _advisor.sanitize_viz_markdown("prosa\n```csv\na,b")
    assert out.count("```") % 2 == 0


def test_persist_viz_audit_usa_separador_estrela_nao_traco():
    # Regressao: o rodape usava '---', que dentro de um {{< include >}} em callout
    # derrubava o render inteiro (YAMLException). Nao voltar atras.
    import inspect
    src = inspect.getsource(_advisor._rodape_viz)   # o rodape mora aqui desde o painel
    assert "***" in src
    assert "\n\n---\n" not in src
    for txt in (_advisor._rodape_viz(["openai/gpt-4o"], plural=False),
                _advisor._rodape_viz(["openai/gpt-4o", "ollama/gemma4:31b"], plural=True)):
        assert txt.startswith("\n\n***\n") and "\n---\n" not in txt


def test_persist_viz_audit_versiona_em_vez_de_sobrescrever(tmp_path):
    # Esses textos passam por revisao humana e vao ao site: reexecutar a auditoria
    # nao pode apagar a versao revisada sem deixar copia.
    run = tmp_path / "youtube_doc_20260722_000000"
    run.mkdir()
    cfg = {"provider": "ollama", "model": "gemma4:31b"}
    _advisor.persist_viz_audit(str(run), "youtube_doc", "stm", {"grid_k": "versao 1"}, "R1", cfg)
    _advisor.persist_viz_audit(str(run), "youtube_doc", "stm", {"grid_k": "versao 2"}, "R2", cfg)
    atual = (run / "advisor_viz" / "grid_k.md").read_text(encoding="utf-8")
    assert "versao 2" in atual
    backups = list((run / "advisor_viz").glob("grid_k.md.bak-*"))
    assert len(backups) == 1 and "versao 1" in backups[0].read_text(encoding="utf-8")


# --- Fase site em painel: as duas leituras no mesmo arquivo ------------------
_PANEL_PARAMS = {
    "advisor": {"provider": "openai", "model": "gpt-4o", "viz_panel": True,
                "fallback": {"provider": "ollama", "model": "gemma4:31b"}},
    "corpora": {"youtube_doc": {"description": "noticias"}},
}


def _fixture_run_lda(tmp_path):
    """(base, run, slugs_com_evidencia) — so os slugs COM CSV recebem prosa; os
    demais viram placeholder deterministico e nao servem para checar leitura."""
    base = tmp_path / "stm"
    run = base / "youtube_doc_20260722_000000"
    run.mkdir(parents=True)
    (run / "stm_results.csv").write_text("post_id,topic_id\n1,0\n", encoding="utf-8")
    (run / "stm_metrics.csv").write_text("k_grid_scores\n1\n", encoding="utf-8")
    (run / "stm_topics_frex.csv").write_text("topic_id,keywords_frex\n0,a b\n",
                                                 encoding="utf-8")
    ev = _advisor.collect_evidence("youtube_doc", "stm", str(base))
    com, _sem = _advisor.evidencia_por_slug(ev, "stm")
    return base, run, [s for s, _, _ in com]


def test_painel_de_leituras_nao_deixa_fallback_dentro_das_cfgs():
    # Se a cfg do painelista carregasse `fallback`, a falha do primary viraria
    # uma 2a chamada ao gemma e a pagina publicaria como "duas leituras
    # independentes" duas respostas do MESMO modelo.
    cfgs = _advisor.painel_de_leituras(_PANEL_PARAMS["advisor"])
    assert [c["model"] for c in cfgs] == ["gpt-4o", "gemma4:31b"]
    assert all("fallback" not in c for c in cfgs)
    # Sem viz_panel, comportamento antigo: uma cfg so (com o fallback dentro).
    sem = dict(_PANEL_PARAMS["advisor"]); sem.pop("viz_panel")
    assert _advisor.painel_de_leituras(sem) == [sem]


def test_run_viz_audit_painel_grava_as_duas_leituras_no_mesmo_arquivo(tmp_path, monkeypatch):
    base, run, slugs = _fixture_run_lda(tmp_path)

    def fake_call(messages, cfg):
        return "\n".join(f"## {s}\nprosa de {cfg['model']}" for s in slugs), cfg

    monkeypatch.setattr(_advisor, "call_advisor", fake_call)
    out = _advisor.run_viz_audit("youtube_doc", "stm", str(base), params=_PANEL_PARAMS)
    assert [c["model"] for c in out["painel"]] == ["gpt-4o", "gemma4:31b"]
    body = (run / "advisor_viz" / f"{slugs[0]}.md").read_text(encoding="utf-8")
    assert "**Leitura A — gpt-4o**" in body and "**Leitura B — gemma4:31b**" in body
    assert "prosa de gpt-4o" in body and "prosa de gemma4:31b" in body
    assert "Duas leituras independentes" in body   # rodape no plural
    led = _advisor.read_ledger(str(base / "llm_advisor_state.json"))["rounds"][-1]
    assert [p["model"] for p in led["painel"]] == ["gpt-4o", "gemma4:31b"]


def test_run_viz_audit_painel_com_um_provider_fora_publica_o_outro(tmp_path, monkeypatch):
    # "E o um?": se so um responde, o arquivo sai com UMA leitura, sem rotulo
    # 'Leitura A', e com o rodape no singular atribuido a quem respondeu.
    base, run, slugs = _fixture_run_lda(tmp_path)

    def fake_call(messages, cfg):
        if cfg["provider"] == "openai":
            raise RuntimeError("429 rate limit")
        return "\n".join(f"## {s}\nso o gemma" for s in slugs), cfg

    monkeypatch.setattr(_advisor, "call_advisor", fake_call)
    out = _advisor.run_viz_audit("youtube_doc", "stm", str(base), params=_PANEL_PARAMS)
    assert [c["model"] for c in out["painel"]] == ["gemma4:31b"]
    assert out["falhas"][0]["model"] == "gpt-4o"
    body = (run / "advisor_viz" / f"{slugs[0]}.md").read_text(encoding="utf-8")
    assert "Leitura A" not in body and "so o gemma" in body
    assert "Texto gerado por IA (ollama/gemma4:31b" in body
    assert "gpt-4o" not in body
    led = _advisor.read_ledger(str(base / "llm_advisor_state.json"))["rounds"][-1]
    assert led["status"] == "partial_panel"
    assert led["painel_falhas"][0]["model"] == "gpt-4o"


def test_run_viz_audit_painel_levanta_se_os_dois_falham(tmp_path, monkeypatch):
    base, _run, _slugs = _fixture_run_lda(tmp_path)

    def fake_call(messages, cfg):
        raise RuntimeError("sem rede")

    monkeypatch.setattr(_advisor, "call_advisor", fake_call)
    with pytest.raises(_advisor.AdvisorConfigError):
        _advisor.run_viz_audit("youtube_doc", "stm", str(base), params=_PANEL_PARAMS)


def test_painel_nao_duplica_placeholder_como_duas_leituras(tmp_path):
    # O placeholder e inserido em CODIGO e sai identico dos dois providers:
    # publicar duas vezes como "leituras independentes" seria falso.
    run = tmp_path / "youtube_doc_20260722_000000"
    run.mkdir()
    secs = {"tsne_theta": _advisor.PLACEHOLDER_SEM_DADOS}
    _advisor.persist_viz_panel(str(run), "youtube_doc", "stm",
                               [({"provider": "openai", "model": "gpt-4o"}, secs),
                                ({"provider": "ollama", "model": "gemma4:31b"}, dict(secs))],
                               "RESP")
    body = (run / "advisor_viz" / "tsne_theta.md").read_text(encoding="utf-8")
    assert body.count(_advisor.PLACEHOLDER_SEM_DADOS) == 1
    assert "Leitura A" not in body
    assert "Texto gerado por IA (openai/gpt-4o e ollama/gemma4:31b" in body


def test_persist_viz_panel_preserva_nota_do_autor_no_rerun(tmp_path):
    # A auditoria humana anota o erro de uma das leituras no proprio arquivo;
    # re-rodar a fase site nao pode republicar o callout SEM a ressalva.
    run = tmp_path / "youtube_doc_20260722_000000"
    run.mkdir()
    cfgs = [{"provider": "openai", "model": "gpt-4o"},
            {"provider": "ollama", "model": "gemma4:31b"}]
    _advisor.persist_viz_panel(str(run), "youtube_doc", "stm",
                               [(cfgs[0], {"grid_k": "v1 gpt"}),
                                (cfgs[1], {"grid_k": "v1 gemma"})], "R1")
    path = run / "advisor_viz" / "grid_k.md"
    nota = "> **Nota do autor (auditoria de 2026-08-01):** a Leitura A erra o pico."
    path.write_text(path.read_text(encoding="utf-8").replace(
        "\n\n***\n", f"\n\n{nota}\n\n***\n", 1), encoding="utf-8")

    _advisor.persist_viz_panel(str(run), "youtube_doc", "stm",
                               [(cfgs[0], {"grid_k": "v2 gpt"}),
                                (cfgs[1], {"grid_k": "v2 gemma"})], "R2")
    novo = path.read_text(encoding="utf-8")
    assert "v2 gpt" in novo and "v1 gpt" not in novo
    assert nota in novo, "nota de auditoria humana sumiu no re-run"
    assert len(list((run / "advisor_viz").glob("grid_k.md.bak-*"))) == 1


def test_nota_do_autor_sobrevive_a_separador_no_meio_da_prosa(tmp_path):
    # Regressao: `sanitize_viz_markdown` converte '---' da prosa do LLM em '***'.
    # Com split() na PRIMEIRA ocorrencia, o corpo era cortado antes da nota e a
    # ressalva de auditoria sumia no re-run, sem erro nenhum.
    viz = tmp_path / "advisor_viz"; viz.mkdir()
    nota = "> **Nota do autor (auditoria):** a Leitura A erra o pico."
    path = viz / "grid_k.md"
    path.write_text(f"Primeira parte.\n\n***\n\nSegunda parte.\n\n{nota}"
                    f"\n\n***\n*rodape*\n", encoding="utf-8")
    assert _advisor._nota_do_autor(str(path)) == nota


def test_nota_do_autor_preserva_nota_de_varias_linhas(tmp_path):
    # Uma ressalva de auditoria costuma ter mais de uma linha; devolver so a
    # primeira descartava o resto silenciosamente no re-run.
    viz = tmp_path / "advisor_viz"; viz.mkdir()
    nota = ("> **Nota do autor (auditoria):** a Leitura A erra o pico.\n"
            "Motivo: confundiu o eixo, ver CSV linha 4.")
    path = viz / "grid_k.md"
    path.write_text(f"Prosa.\n\n{nota}\n\n***\n*rodape*\n", encoding="utf-8")
    assert _advisor._nota_do_autor(str(path)) == nota


def test_painel_nao_marca_como_faltante_slug_que_um_provider_entregou(tmp_path, monkeypatch):
    # Regressao: `missing` usava a UNIAO das faltas por provider, entao um slug
    # respondido por A e nao por B aparecia em viz_written E em viz_missing ao
    # mesmo tempo — o ledger contradizia o arquivo publicado.
    base, run, slugs = _fixture_run_lda(tmp_path)
    assert len(slugs) >= 2
    orfao = slugs[0]

    def fake_call(messages, cfg):
        if cfg["provider"] == "ollama":       # gemma nao entrega o 1o slug
            entrega = [s for s in slugs if s != orfao]
        else:
            entrega = slugs
        return "\n".join(f"## {s}\nprosa {cfg['model']}" for s in entrega), cfg

    monkeypatch.setattr(_advisor, "call_advisor", fake_call)
    out = _advisor.run_viz_audit("youtube_doc", "stm", str(base), params=_PANEL_PARAMS,
                                 retry_on_missing=False)
    assert out["missing"] == [], "slug entregue por um provider nao e faltante"
    body = (run / "advisor_viz" / f"{orfao}.md").read_text(encoding="utf-8")
    assert "prosa gpt-4o" in body and "Leitura A" not in body
    led = _advisor.read_ledger(str(base / "llm_advisor_state.json"))["rounds"][-1]
    assert led["status"] == "audited" and led["viz_missing"] == []
    # a lacuna do painelista continua rastreada, sem marcar o arquivo como ausente
    assert led["painel_lacunas"]["ollama/gemma4:31b"] == [orfao]


# --- Fatos derivados dos slugs que a auditoria 2026-08-01 pegou --------------
def test_fatos_derivados_heatmap_matriz_da_top_da_linha_e_da_coluna():
    # Os 2 piores erros do gpt (auditoria 2026-08-01) foram superlativos numa
    # matriz beta sem fato pronto: "T0 tem maior peso para policia" quando o
    # maior da LINHA era outro e o maior da COLUNA estava noutro topico.
    beta = ("topic_id,brasilia,ser,policia,federal\n"
            "0,0.0324,0.0318,0.0203,0.0188\n"
            "2,0.0,0.0,0.0585,0.0593\n")
    out = _advisor._fatos_derivados("heatmap_phi", "nmf", {"nmf_beta.csv": beta})
    assert "T0: brasilia=0,0324" in out            # maior da linha, nao 'policia'
    assert "policia: maior em T2 (0,0585)" in out  # maior da coluna
    assert "maior peso da tabela: T2xfederal=0,0593" in out
    assert "NÃO afirme peso baixo nem ausência" in out   # aviso de truncamento


def test_fatos_derivados_keywords_expoe_pares_que_compartilham_termo():
    # O erro do gemma foi afirmar que T1 e T5 nao compartilhavam termos com os
    # demais — contradito pelo proprio CSV entregue.
    kw = ("topic_id,topic_name,keywords,model\n"
          "1,Espaco,\"espacial, lua, terra\",lda\n"
          "2,Ambiente,\"governo, terra, brasil\",lda\n"
          "5,Futebol,\"jogo, futebol, brasileiro\",lda\n"
          "19,Olimpicos,\"atleta, futebol, brasileiro\",lda\n")
    out = _advisor._fatos_derivados("heatmap_phi", "stm", {"stm_topics_for_eval.csv": kw})
    assert "T1 e T2: terra" in out
    assert "T5 e T19: brasileiro, futebol" in out
    # e a lista de termos de topico unico vem com a ressalva anti-"exclusivo"
    assert "Proibido escrever 'exclusivo'" in out


def test_fatos_derivados_grid_k_sinaliza_serie_nao_monotonica():
    # "C_v aumenta progressivamente ate K=25" passou porque nao havia fato algum
    # para grid_k e a evidencia e um dict JSON dentro de uma celula de CSV.
    metrics = ('model,k_grid_scores\n'
               'nmf,"{""3"": 0.4285, ""7"": 0.5571, ""8"": 0.5364, ""25"": 0.6093}"\n')
    out = _advisor._fatos_derivados("grid_k", "nmf", {"nmf_metrics.csv": metrics})
    assert "melhor C_v do grid: 0,6093 em K=25" in out
    assert "NÃO é monotônica" in out
    assert "K=7 (0,5571) para K=8 (0,5364)" in out
    assert "0,4285" in out and "0.4285" not in out   # virgula decimal, via _br


def test_fatos_derivados_grid_k_reconhece_serie_monotonica():
    metrics = ('model,k_grid_scores\n'
               'lda,"{""3"": 0.30, ""5"": 0.40, ""10"": 0.55}"\n')
    out = _advisor._fatos_derivados("grid_k", "stm", {"stm_metrics.csv": metrics})
    assert "é monotônica neste grid" in out and "NÃO é monotônica" not in out


def test_fatos_derivados_grid_k_deriva_npmi_e_diversity_e_avisa_divergencia():
    # Protocolo pos-22/08/2026 (docs/protocolo_selecao.md §1.5/§1.6): NPMI decide
    # o K (Pareto NPMI x Diversity), C_v so reporta. k_npmi_scores/k_diversity_scores
    # ja existem em nmf_metrics.csv/stm_metrics.csv reais mas nao tinham fato
    # derivado nenhum ate esta mudanca.
    metrics = ('model,k_grid_scores,k_npmi_scores,k_diversity_scores\n'
               'nmf,"{""3"": 0.40, ""7"": 0.55, ""8"": 0.53, ""25"": 0.60}",'
               '"{""3"": 0.01, ""7"": 0.02, ""8"": 0.09, ""25"": 0.05}",'
               '"{""3"": 0.70, ""7"": 0.75, ""8"": 0.80, ""25"": 0.60}"\n')
    out = _advisor._fatos_derivados("grid_k", "nmf", {"nmf_metrics.csv": metrics})
    assert "melhor NPMI do grid: 0,0900 em K=8" in out
    assert "melhor Diversity do grid: 0,8000 em K=8" in out
    assert "melhor C_v do grid: 0,6000 em K=25" in out
    # K de melhor NPMI (8) diverge do K de melhor C_v (25) -- tem que avisar.
    assert "ATENÇÃO: o K de melhor NPMI diverge do K de melhor C_v" in out


def test_fatos_derivados_grid_k_sem_colunas_novas_nao_quebra():
    # CSV legado (pre-22/08/2026): so k_grid_scores, sem k_npmi_scores/
    # k_diversity_scores. Nao pode quebrar nem inventar fato para coluna ausente
    # (D3 do PRD 2026-08-28: degradar sem quebrar).
    metrics = ('model,k_grid_scores\n'
               'nmf,"{""3"": 0.40, ""7"": 0.55, ""8"": 0.53, ""25"": 0.60}"\n')
    out = _advisor._fatos_derivados("grid_k", "nmf", {"nmf_metrics.csv": metrics})
    assert "melhor C_v do grid: 0,6000 em K=25" in out
    assert "NPMI" not in out
    assert "Diversity" not in out
    assert "ATENÇÃO" not in out


def test_bh_ajustar_reproduz_exemplo_canonico():
    # Exemplo classico de Benjamini & Hochberg (1995).
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    esperado = [0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.10571, 0.216, 0.216, 0.216]
    obtido = _advisor._bh_ajustar(p)
    assert all(abs(a - b) < 1e-4 for a, b in zip(obtido, esperado))


def test_derive_stm_prevalence_corrige_multiplicidade(tmp_path):
    # ~225 testes (25 topicos x 9 categorias): a alpha=0,05 sem correcao, ~11
    # falsos positivos entram na lista rotulada "significativos".
    import pandas as pd
    n = 225
    df = pd.DataFrame({
        "topic_id": list(range(n)),
        "term": ["categoryesporte"] * n,
        "estimate": [0.1] * n,
        # 1 efeito forte + 224 p-valores espalhados entre 0,01 e 0,99: sem BH,
        # dezenas passam; com BH, so o forte sobrevive.
        "p_value": [0.00001] + [0.01 + i * (0.98 / 224) for i in range(n - 1)],
    })
    p = tmp_path / "stm_prevalence_effects.csv"
    df.to_csv(p, index=False)
    out = _advisor._derive_stm_prevalence_summary(str(p))
    assert "p_ajustado_bh" in out
    assert len(out.strip().splitlines()) - 1 == 1        # so o efeito forte


def test_load_results_exclui_outlier_renumerado_do_bertopic(tmp_path):
    # folha mantem outliers em -1, mas tweets_bre2022 renumera o bucket residual
    # para o ultimo id com topic_name "Outlier". Filtrar so -1 deixava 1452 docs
    # de outlier entrarem no cross-tab como se fossem um topico real.
    res = tmp_path / "bertopic_results.csv"
    res.write_text(
        "post_id,topic_id,topic_name,granularity\n"
        "p1,0,Tema A,unit\n"
        "p2,24,Outlier,unit\n"
        "p3,-1,Topic_-1,unit\n", encoding="utf-8")
    cor = tmp_path / "corpus_limpo.csv"
    cor.write_text("post_id,category\np1,poder\np2,poder\np3,poder\n", encoding="utf-8")
    df = _advisor._load_results_with_corpus(str(res), str(cor))
    assert sorted(df["topic_id"]) == [0]


# --- Fatos derivados: tira do LLM a aritmetica comparativa -------------------
# Cada teste abaixo codifica um erro REAL publicado na auditoria de 2026-08-01.
# O padrao dos 8 erros era sempre o mesmo: valor individual certo, relacao entre
# valores errada (superlativo, ordinal, pareamento id<->valor).

def test_fatos_derivados_cross_tab_acerta_concentrada_e_dispersa():
    # Erro real (lda-folha/topico_categoria): chamou de "maior dispersao" a
    # linha cuja maior fatia era 35,3% — a dispersa era a de 25,4%. Concentracao
    # e o MAXIMO da linha; dispersao e o oposto.
    csv = ("category,n_docs,T0,T1,T12\n"
           "equilibrioesaude,499,1.2,0.0,66.7\n"
           "cotidiano,498,2.8,0.2,35.3\n"
           "ciencia,484,0.8,25.4,1.7\n")
    out = _advisor._fatos_derivados("topico_categoria", "stm",
                                    {"derived_topic_category.csv": csv})
    assert "equilibrioesaude" in out.split("mais DISPERSA")[0]
    assert "ciencia" in out.split("mais DISPERSA")[1].splitlines()[0]
    assert "cotidiano" not in out.split("mais DISPERSA")[1].splitlines()[0]


def test_fatos_derivados_cross_tab_pareia_id_com_valor():
    # Erro real (bertopic-folha/topico_categoria): valores certos, IDs deslizados
    # uma casa (T12 citado como T13). Os pares tem que sair prontos.
    csv = ("category,n_docs,T0,T12,T16\n"
           "mundo,337,22.3,23.4,11.3\n")
    out = _advisor._fatos_derivados("topico_categoria", "bertopic",
                                    {"derived_topic_category.csv": csv})
    assert "T12=23,4%" in out
    assert "T16=11,3%" in out
    assert "T13" not in out and "T17" not in out


def test_fatos_derivados_ranking_ignora_ordem_do_arquivo():
    # Erro real (lda-tweets/docs_por_topico): o *_exclusividade_ranking.csv vem
    # ordenado por EXCLUSIVIDADE; o modelo leu de cima para baixo e apresentou
    # T12 (615) como 3o maior em volume, pulando T0 (619).
    csv = ("topic_id,topic_name,exclusividade,n_docs\n"
           "15,Mane,0.803,555\n"
           "12,Nordeste,0.727,615\n"
           "18,Globolixo,0.574,729\n"
           "11,Diplomacao,0.495,714\n"
           "0,Debates,0.451,619\n")
    out = _advisor._fatos_derivados("docs_por_topico", "stm",
                                    {"stm_exclusividade_ranking.csv": csv})
    volume = out.split("por n_docs")[1]
    # a ordem correta e 729 > 714 > 619 > 615: T0 vem ANTES de T12
    assert volume.index("T0") < volume.index("T12")
    assert "T18" in volume.split("T11")[0]


def test_fatos_derivados_grid_acha_o_minimo_de_perplexidade():
    # Erro real (investigacao-lda/grid_priors): disse que 3302,77 era a menor
    # perplexidade do grid; a menor era 3295,30, em outra linha.
    csv = ("alpha,eta,cv,perplexity\n"
           "symmetric,0.01,0.510088,3979.811664\n"
           "0.01,0.10,0.500182,3302.773534\n"
           "asymmetric,0.10,0.497078,3295.304127\n")
    out = _advisor._fatos_derivados("grid_priors", "stm",
                                    {"stm_alpha_eta_grid.csv": csv})
    # separador de milhar tambem em convencao PT-BR: 3.295,30
    assert "3.295,30" in out
    assert "3.302,77" not in out.split("menor perplexity")[1]
    assert "0,5101" in out
    # CSV legado sem NPMI (pre-22/08/2026) -- nao pode inventar o fato.
    assert "maior NPMI" not in out


def test_fatos_derivados_grid_hparams_npmi_maiusculo_nmf_e_cobertura_ativa():
    # Casing do braco NMF e "NPMI" maiusculo (nmf_kappa_minprob_grid.csv real).
    # Protocolo §1.6: cobertura e porta CONDICIONAL -- so decide se o minimo da
    # grade cair abaixo de ~0,999 (o mesmo limiar de
    # _selecao.py::_porta_cobertura_nmf, para a prosa nunca discordar do pipeline).
    csv = ("kappa,minimum_probability,cv,NPMI,Diversity,cobertura,k,n_docs_bow_vazio\n"
           "1.0,0.0,0.60,0.10,0.80,1.0,15,27\n"
           "2.0,0.0,0.62,0.12,0.75,0.97,15,27\n"
           "3.0,0.0,0.45,0.03,0.20,0.94,15,27\n")
    out = _advisor._fatos_derivados("grid_hparams", "nmf",
                                    {"nmf_kappa_minprob_grid.csv": csv})
    assert "maior NPMI do grid: 0,1200 em [" in out
    assert "kappa=2.0" in out.split("maior NPMI do grid")[1].splitlines()[0]
    assert "cobertura NÃO é ~100% em toda a grade: mínimo 0,9400" in out
    assert "kappa=3.0" in out.split("mínimo 0,9400")[1].splitlines()[0]
    assert "porta ATIVA" in out
    assert "27 documento(s) com BOW vazio" in out


def test_fatos_derivados_grid_priors_npmi_minusculo_stm_sem_cobertura():
    # Casing do braco LDA (alpha/eta) e "npmi" minusculo (stm_alpha_eta_grid.csv
    # real); esse grid nao tem coluna cobertura -- nao pode quebrar por isso.
    csv = ("alpha,eta,cv,npmi,perplexity\n"
           "symmetric,0.01,0.51,0.08,3979.81\n"
           "asymmetric,0.10,0.49,0.11,3295.30\n")
    out = _advisor._fatos_derivados("grid_priors", "stm",
                                    {"stm_alpha_eta_grid.csv": csv})
    assert "maior NPMI do grid: 0,1100 em [" in out
    assert "cobertura" not in out
    assert "porta" not in out


def test_fatos_derivados_grid_hparams_cobertura_vacua_nao_alarma():
    # Cobertura uniformemente ~100% (nenhum ponto < 0,999): a porta e VACUA
    # (protocolo §1.6.2) e o fato nao deve soar como alarme.
    csv = ("kappa,minimum_probability,cv,NPMI,Diversity,cobertura,k\n"
           "1.0,0.0,0.60,0.10,0.80,1.0,15\n"
           "2.0,0.0,0.62,0.12,0.75,0.9995,15\n")
    out = _advisor._fatos_derivados("grid_hparams", "nmf",
                                    {"nmf_kappa_minprob_grid.csv": csv})
    assert "porta VÁCUA" in out
    assert "porta ATIVA" not in out
    assert "BOW vazio" not in out   # coluna n_docs_bow_vazio ausente deste CSV


def test_fatos_derivados_devolve_vazio_sem_csv_conhecido():
    assert _advisor._fatos_derivados("similaridade_topicos", "stm", {}) == ""
    assert _advisor._fatos_derivados("topico_categoria", "stm",
                                     {"coisa_qualquer.csv": "a,b\n1,2\n"}) == ""


def test_evidencia_por_slug_injeta_bloco_de_fatos():
    csv = ("category,n_docs,T0,T1\n"
           "poder,100,85.4,14.6\n"
           "mundo,50,40.0,60.0\n")
    ev = {"csv_snippets": {"derived_topic_category.csv": csv}}
    com, _sem = _advisor.evidencia_por_slug(ev, "stm")
    texto = dict((s, t) for s, _ti, t in com)["topico_categoria"]
    assert "FATOS DERIVADOS" in texto
    assert "derived_topic_category.csv" in texto     # o CSV cru continua junto


def test_modelos_de_raciocinio_reconhecidos():
    for m in ("gpt-5-mini", "gpt-5.4-nano", "GPT-5", "o3-mini", "o4-mini"):
        assert _advisor._e_modelo_de_raciocinio(m), m
    for m in ("gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "gemma4:31b", "", None):
        assert not _advisor._e_modelo_de_raciocinio(m), m


def test_call_one_usa_parametros_certos_por_familia(monkeypatch):
    # gpt-5* recusa max_tokens/temperature; mandar os antigos da 400, o 400 vira
    # excecao e o fallback (Ollama) responde no lugar — silenciosamente.
    vistos = {}

    class _Resp:
        choices = [type("C", (), {"message": type("M", (), {"content": "x"})()})()]

    class _Cli:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    vistos.clear(); vistos.update(kw); return _Resp()

    monkeypatch.setattr(_advisor, "_build_client", lambda cfg: _Cli())

    _advisor._call_one([], {"model": "gpt-5-mini", "max_output_tokens": 4000})
    assert "max_completion_tokens" in vistos and "max_tokens" not in vistos
    assert "temperature" not in vistos
    assert vistos["max_completion_tokens"] >= 8000   # folga p/ tokens de raciocinio

    _advisor._call_one([], {"model": "gpt-4o", "max_output_tokens": 4000})
    assert vistos["max_tokens"] == 4000 and vistos["temperature"] == 0.2
    assert "max_completion_tokens" not in vistos


# --- Auditoria de gastos ------------------------------------------------------
def test_contar_tokens_usa_tiktoken_e_nao_quebra_sem_ele():
    n = _advisor.contar_tokens("uma frase curta em portugues", "gpt-4.1-mini")
    assert isinstance(n, int) and 3 < n < 20
    assert _advisor.contar_tokens("", "gpt-4.1-mini") == 0
    # modelo desconhecido nao pode levantar — cai num encoding padrao
    assert _advisor.contar_tokens("abc def", "modelo-que-nao-existe:9b") > 0


def test_custo_estimado_usa_tabela_de_precos():
    # gpt-4.1-mini: 0,40 entrada / 1,60 saida por 1M
    c = _advisor.custo_estimado("gpt-4.1-mini", entrada=1_000_000, saida=1_000_000)
    assert abs(c - 2.00) < 1e-6
    # cached input e cobrado mais barato quando informado
    c2 = _advisor.custo_estimado("gpt-4.1-mini", entrada=1_000_000, saida=0,
                                 entrada_cacheada=1_000_000)
    assert abs(c2 - 0.10) < 1e-6
    # provider local nao tem preco -> custo zero, nunca excecao
    assert _advisor.custo_estimado("gemma4:31b", entrada=999, saida=999) == 0.0


def test_registrar_uso_grava_jsonl_append(tmp_path):
    log = tmp_path / "advisor_usage.jsonl"
    for i in range(3):
        _advisor.registrar_uso(
            {"provider": "openai", "model": "gpt-4.1-mini"},
            entrada=1000 * (i + 1), saida=100, entrada_cacheada=0, raciocinio=0,
            contexto={"corpus": "youtube_doc", "modelo_topico": "stm", "fase": "site"},
            path=str(log))
    linhas = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(linhas) == 3
    assert linhas[0]["corpus"] == "youtube_doc" and linhas[0]["fase"] == "site"
    assert linhas[2]["entrada"] == 3000
    assert all("timestamp" in l and "custo_usd" in l for l in linhas)


def test_relatorio_de_gastos_agrega_por_modelo_e_rodada(tmp_path):
    log = tmp_path / "advisor_usage.jsonl"
    for mod, ent in (("gpt-4.1-mini", 10_000), ("gpt-4.1-mini", 20_000),
                     ("gemma4:31b", 50_000)):
        _advisor.registrar_uso({"provider": "openai", "model": mod},
                               entrada=ent, saida=1000, entrada_cacheada=0,
                               raciocinio=0,
                               contexto={"corpus": "youtube_doc", "modelo_topico": "stm",
                                         "fase": "site"},
                               path=str(log))
    rel = _advisor.relatorio_de_gastos(str(log))
    assert "gpt-4.1-mini" in rel and "gemma4:31b" in rel
    assert "30.000" in rel or "30,000" in rel or "30000" in rel   # entrada somada
    assert "TOTAL" in rel.upper()


def test_relatorio_de_gastos_sem_log_nao_quebra(tmp_path):
    rel = _advisor.relatorio_de_gastos(str(tmp_path / "nao_existe.jsonl"))
    assert isinstance(rel, str) and rel.strip()


def test_fatos_derivados_frex_avisa_que_nao_ha_coluna_numerica():
    # 2a medicao da auditoria (2026-08-01): o gpt citou "bolsonaro (peso 0,761)"
    # a partir de um CSV que so tem id, nome e lista de termos. O aviso tem que
    # vir junto do numero — no bloco de fatos, nao so no glossario.
    frex = ("topic_id,topic_name,keywords_frex\n"
            "0,Manifestacao,\"mane, perdeu, barroso\"\n"
            "1,Bolsonarismo,\"trabalha, bolsonaro, live\"\n")
    out = _advisor._fatos_derivados("frex", "stm", {"stm_topics_frex.csv": frex})
    assert "NÃO tem nenhuma coluna numérica" in out
    assert "PROIBIDO citar peso" in out
    assert "Proibido escrever 'exclusivo'" in out


def test_fatos_derivados_frex_nao_avisa_quando_ha_coluna_numerica():
    # O aviso e condicional: se a evidencia tiver peso de verdade, citar peso e
    # legitimo e o aviso seria falso.
    com_peso = ("topic_id,topic_name,keywords,score\n"
                "0,Tema,\"a, b\",0.5\n1,Outro,\"c, d\",0.7\n")
    out = _advisor._fatos_derivados("keywords", "stm",
                                    {"stm_topics_for_eval.csv": com_peso})
    assert "NÃO tem nenhuma coluna numérica" not in out


def test_fatos_derivados_keywords_traz_contagem_por_termo():
    # 2a medicao: "bolsonaro em pelo menos 15 topicos" quando eram 12 — o modelo
    # contou de cabeca porque a contagem nao estava pronta.
    kw = ("topic_id,topic_name,keywords,model\n"
          "0,A,\"bolsonaro, lula\",lda\n"
          "1,B,\"bolsonaro, brasil\",lda\n"
          "2,C,\"bolsonaro, terra\",lda\n"
          "3,D,\"pesquisa\",lda\n")
    out = _advisor._fatos_derivados("heatmap_phi", "stm", {"stm_topics_for_eval.csv": kw})
    assert "bolsonaro: 3 tópicos (T0, T1, T2)" in out
    assert "em quantos tópicos desta lista cada termo repetido aparece" in out
