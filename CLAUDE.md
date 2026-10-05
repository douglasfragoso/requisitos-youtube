# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research pipeline: YouTube product-review video transcripts → STM at document level and
global NMF followed by local NMF at sentence level → candidate requirements for review. The
historical STM-sentence to NMF pipeline is a baseline, not the final sentence model. Current
canonical design doc: `docs/idealizacao-artigo.txt` (actively maintained). Literature indexes: `articles/artigos_mapeados.txt`,
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
03-topic-modeling/  STM document + global/local NMF sentence; STM sentence is a baseline
04-requirements/    joins NMF topic output into a human-review evidence table; NEVER auto-labels a
                     requirement -- requirement_candidate/review_decision start blank and are only
                     ever filled in by a human, for evidence of any sentiment polarity
```

Each stage is independently testable. `01-preprocessing` has its own `configs/params.yaml`;
everything from `03-topic-modeling` onward shares `03-topic-modeling/configs/params.yaml`.

### `03-topic-modeling`: final NMF sentence pipeline and STM document analysis

`_helpers.py` (~2.9k lines, the single most load-bearing file in the repo) prepares STM input,
invokes `run_stm.R` as a subprocess, and re-scores models in Python. The global sentence NMF
has its own script and does not invoke the STM engine.

**K-selection metric is scoped per model family — this is easy to get backwards:**
- **STM document**: the pinned corrected run uses K=12, selected by peak C_v in its notebook;
  `docs/idealizacao-artigo.txt` records the unresolved difference from the planned Pareto rule.
- **Global sentence NMF**: final K=20 was chosen after qualitative inspection of K=15 and K=20;
  reconstruction error alone was not treated as evidence of elicitation quality.
- **Local sentence NMF**: K=2–6 for each selected global topic, Pareto frontier of NPMI ×
  diversity, then highest NPMI on that frontier. C_v is reported, not used to choose K.
- Admissible K ranges and `no_below`/`no_above` differ per corpus (`youtube_doc` vs
  `youtube_sent`) and live in `params.yaml`; never hardcode a K range in notebook code.

**LLM config in `params.yaml`:**
- `llm`: names STM/NMF topics and subtopics (`name_all_topics` in `_helpers.py`). Output is **not
  deterministic** — always cite the run directory a name came from, never assume it's stable
  across reruns. Has a documented failure mode of plausible-sounding but wrong labels (e.g. a
  topic actually about camera reviews named after an unrelated keyword it happened to score
  highest on) — verify a generated name against a sample of the underlying documents before using
  it in analysis or prose.

**Final local NMF** (`03-topic-modeling/scripts/run_restricted_nmf_gensim.py`) consumes selected
topics from the pinned global NMF run. `04-requirements/nmf_requirement_review.py` joins global
and local results by `sent_id`, validates their original text/category/topic/weight, and creates
the final blinded human-review sample. The final configuration is
`params.yaml::final_sentence_pipeline`. The old notebook
`03-topic-modeling/notebooks/nmf/01_nmf_youtube_sent_aspecto.ipynb` uses positional alignment
with STM's `theta`; it is only for the historical baseline.

### Output data is not versioned

`.gitignore` excludes `*/data/raw/*`, `*/data/output/*`, `docs/`, and `articles/`. The existing
`docs/idealizacao-artigo.txt` is already tracked and remains versioned; new analysis reports and
model runs are local unless explicitly added. The exception to the blanket `*.json` ignore is
`*/configs/*.json`. The final run IDs and selected topics are pinned in `params.yaml`, while the
untracked run directories hold their own detailed manifests and data.
