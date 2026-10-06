# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research pipeline: YouTube product-review video transcripts → global NMF followed by local NMF at
sentence level → candidate requirements for human review. STM, Guided NMF, BRETT and the
STM-sentence baseline were removed in the 2026-10 cleanup (see git history before commit
`2c10749`). Current canonical design doc: `docs/idealizacao-artigo.txt` (actively maintained; it
still describes STM sections that predate the cleanup). Literature indexes:
`articles/artigos_mapeados.txt`, `docs/related-works.md`.

## Commands

Python 3.12 venv at `.venv/`; install with `pip install -r requirements.txt` then
`python -m spacy download en_core_web_sm`.

Run the test suites per stage (no root pytest config):

```
.venv/Scripts/python -m pytest 00-dataset 01-preprocessing 02-sentences 03-topic-modeling 04-requirements
```

Tests never fit the real NMF. Verify model runs by executing the notebooks and inspecting their
output and the run folders (`grid.csv`, `manifest.json`).

Notebooks run via `jupyter nbconvert --to notebook --execute --inplace <path>` (kernel
`topicmodeling-venv`). `03-topic-modeling/scripts/run_notebook_detached.ps1` launches a notebook
through a Windows Scheduled Task so a long run survives the terminal closing.

## Architecture

### Pipeline stages (numbered, sequential dependency)

```
00-dataset/         raw JSON -> corpus CSV, product-category detection (regex + manual override)
01-preprocessing/   language filter, cleaning -> corpus_limpo.csv
02-sentences/       sentence segmentation with context; punctuation restoration for auto-captions only
03-topic-modeling/  global NMF (sklearn, TF-IDF) -> local NMF (Gensim, lemma BoW) per selected topic
04-requirements/    joins NMF topic output into a human-review evidence table; NEVER auto-labels a
                     requirement -- requirement_candidate/review_decision start blank and are only
                     ever filled in by a human, for evidence of any sentiment polarity
```

Each stage is independently testable. `01-preprocessing` has its own `configs/params.yaml`;
`03-topic-modeling` and `04-requirements` share `03-topic-modeling/configs/params.yaml`.

### `03-topic-modeling`

Run the stages through `notebooks/nmf/01_nmf_global.ipynb` and `02_nmf_topicos.ipynb`; they call
`scripts/run_global_nmf_sentences.py` and `scripts/run_restricted_nmf_gensim.py` (helpers in
`scripts/nmf_utils.py`). The final configuration is `params.yaml::final_sentence_pipeline`.

**K selection:**
- **Global NMF**: K=20 was chosen after qualitative inspection of K=15 and K=20; reconstruction
  error alone was not treated as evidence of elicitation quality.
- **Local NMF**: K=2–6 per selected global topic, Pareto frontier of NPMI × diversity, then highest
  NPMI on that frontier (in practice, the argmax of NPMI). C_v is reported, not used to choose K.
- Ranges and `no_below`/`no_above` live in `params.yaml`; never hardcode them in notebook code.

**Reproducibility:** with seed 42 the full rerun reproduced the paper's numbers (116,111 filtered
sentences, 3,542 terms, local K `[3,4,2,3,4,3,6,6,5,6]`, 57,952 selected sentences). Topic IDs
only stay valid for the pinned `global_run`; if the corpus or vocabulary changes, redo the topic
selection in notebook 01.

`04-requirements/nmf_requirement_review.py` joins global and local results by `sent_id`, validates
their original text/category/topic/weight, applies the fixed request/complaint lexicon, and builds
the rankings and the blinded human-review sample. The review output folder is never overwritten.

### Output data is not versioned

`.gitignore` excludes `*/data/raw/*`, `*/data/output/*`, `docs/`, and `articles/`. The existing
`docs/idealizacao-artigo.txt` is already tracked and remains versioned; new analysis reports and
model runs are local unless explicitly added. The transcript dataset
`00-dataset/transcricoes_youtube_metadados.json` is versioned. The exception to the blanket
`*.json` ignore is `*/configs/*.json`. The final run ID and selected topics are pinned in
`params.yaml`, while the untracked run directories hold their own manifests and data.
