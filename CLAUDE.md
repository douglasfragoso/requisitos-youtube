# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research pipeline: YouTube product-review video transcripts → Structural Topic Model (STM) at two
granularities (document, sentence) → candidate requirements for requirements engineering. Current
canonical design doc: `docs/idealizacao-artigo.txt` (actively maintained; treat it over
`README.md`'s RQ table where they disagree — README references a `docs/specs/...` design doc that
is not present on disk). Literature indexes: `articles/artigos_mapeados.txt`,
`docs/related-works.md`.

## Commands

Python 3.12 venv at `.venv/`; install with `pip install -r requirements.txt` then
`python -m spacy download en_core_web_sm`. R ≥4.4 with the `stm`, `jsonlite`, `glmnet` packages
must be on `PATH` (`Rscript`) — the STM engine runs as an R subprocess
(`03-topic-modeling/scripts/run_stm.R`), it is not called as a Python library.

No root pytest config — each pipeline stage is its own test target:

```
.venv/Scripts/python -m pytest 00-dataset/test_build_corpus.py
.venv/Scripts/python -m pytest 01-preprocessing/test_corpus_limpo.py
.venv/Scripts/python -m pytest 02-sentences/
.venv/Scripts/python -m pytest 03-topic-modeling/notebooks/test_helpers.py   # largest, most load-bearing suite
.venv/Scripts/python -m pytest 04-requirements/test_build_review_table.py
```

Single test: `-k <name>` or `path::test_name`. `test_helpers.py` deliberately does not cover
anything that requires fitting a model (grid searches, `train_*`, sweeps) — those take hours and
depend on corpus state; verify them by running the actual notebook and inspecting its output
(`*_grid_k_diagnostics.csv`, `*_final.json`), not pytest.

Notebooks run via `jupyter nbconvert --to notebook --execute --inplace <path>` (kernel
`topicmodeling-venv`). Long grid sweeps should go through
`03-topic-modeling/scripts/run_notebook_detached.ps1` instead of a plain terminal run — it
launches through a Windows Scheduled Task so the run survives the terminal/session closing (a
plain `nbconvert` run has died silently mid-sweep when its launching terminal closed, with no
corresponding OS crash/sleep event).

## Architecture

### Pipeline stages (numbered, sequential dependency)

```
00-dataset/         raw JSON -> corpus CSV, product-category detection (regex + manual override)
01-preprocessing/   language filter, cleaning -> corpus_limpo.csv
02-sentences/       sentence segmentation with context; punctuation restoration for auto-captions only
03-topic-modeling/  STM (document + sentence) + NMF restrito -- see below, this is where the complexity is
04-requirements/    joins topic output into a human-review evidence table; NEVER auto-labels a
                     requirement -- requirement_candidate/review_decision start blank and are only
                     ever filled in by a human, for evidence of any sentiment polarity
```

Each stage is independently testable. `01-preprocessing` has its own `configs/params.yaml`;
everything from `03-topic-modeling` onward shares `03-topic-modeling/configs/params.yaml`.

### `03-topic-modeling`: Python owns the protocol, R is only the training engine

`_helpers.py` (~2.9k lines, the single most load-bearing file in the repo) prepares STM input,
invokes `run_stm.R` as a subprocess, and re-scores everything in Python — R's own reported
coherence is not what selection decisions are made on.

**K-selection metric is scoped per model family — this is easy to get backwards:**
- **STM**: Pareto frontier of *semantic coherence × exclusivity* (the `stm` package's own
  `searchK` convention, UMass-family coherence) plus diversity. Not NPMI, not C_v.
- **NMF / LDA (including NMF restrito)**: NPMI + diversity *decide* (`_selecao.py`'s `_pareto`);
  C_v is computed and plotted but is reporting-only, never the selection criterion. This was a
  deliberate, measured choice — C_v anticorrelates with NPMI at sentence granularity in this
  project's own grid — not an oversight. Don't switch the criterion without re-measuring on the
  actual corpus in question.
- Admissible K ranges and `no_below`/`no_above` differ per corpus (`youtube_doc` vs
  `youtube_sent`) and live in `params.yaml`; never hardcode a K range in notebook code.

**Two independent LLM configs in `params.yaml` — do not conflate them:**
- `llm`: names STM/NMF topics and subtopics (`name_all_topics` in `_helpers.py`). Output is **not
  deterministic** — always cite the run directory a name came from, never assume it's stable
  across reruns. Has a documented failure mode of plausible-sounding but wrong labels (e.g. a
  topic actually about camera reviews named after an unrelated keyword it happened to score
  highest on) — verify a generated name against a sample of the underlying documents before using
  it in analysis or prose.
- `advisor`: `_advisor.py`, a hyperparameter-calibration assistant (OpenAI primary, Ollama
  fallback). It only *proposes* `params.yaml` patches; it never writes the file or triggers a
  sweep itself.

**NMF restrito** (`03-topic-modeling/notebooks/nmf/01_nmf_youtube_sent_aspecto.ipynb`)
reconstructs its per-topic sentence subset by *positional alignment* with the STM's `theta`
output, guarded by an explicit `assert len(df_full) == len(stm_input)` — there is no reliable
sentence-level join key in `stm_input.csv` (only the video-level `post_id`). Any change to
pre-STM filtering upstream breaks this alignment.

### Output data is not versioned

`.gitignore` excludes `*/data/raw/*`, `*/data/output/*`, `docs/`, and `articles/` — model runs,
metrics CSVs, working specs, and downloaded papers exist only on disk, never in git history. The
one exception to the blanket `*.json` ignore is `*/configs/*.json`. `params.yaml` is therefore the
actual source of truth for what a given run used, since the run's own output directory isn't
committed.
