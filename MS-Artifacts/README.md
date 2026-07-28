# Stylometric Detection of AI-Generated Phishing

Artefact for CO4011 Master's Project — *Stylometric Detection of AI-Generated
Phishing: Distinguishing Known AI-Generated Credential-Harvesting Emails from
Real-World Phishing.*

**Research question.** To what extent can stylometric and lexical features enable
machine-learning classifiers to distinguish known AI-generated
credential-harvesting phishing emails from real-world (Nazario) phishing emails,
and which features are most discriminative?

This directory is the reproducible pipeline that produces every number, table and
figure in the dissertation, from the raw source archives through to the finished
document.

**Class labels.** The two classes are named for what the sources actually
guarantee:

| label | class | source |
|---|---|---|
| `0` | Real-world phishing | Nazario phishing corpus 2022–2024 |
| `1` | Known AI-generated phishing | Eze & Shamir (2024) DeepAI-generated corpus |

The Nazario corpus is hand-classified as phishing but **not** verified for
authorship, so class 0 is *not* described as "human-written" anywhere in this
project.

---

## What this does

```
data/raw/  ->  parse, clean, screen, de-duplicate, balance  ->  data/emails.csv
           ->  stylometric features  ->  train  ->  evaluate  ->  interpret
           ->  robustness checks  ->  validate  ->  dissertation
```

- **Corpus construction** (`scripts/build_corpus.py`): parses the raw mbox and
  text archives, extracts body text only, applies a documented
  credential-harvesting screen, removes exact duplicates and balances the classes.
  Records a per-stage funnel and SHA-256 provenance in `data/corpus_summary.json`.
- **Feature extraction** (`src/features.py`): **95** interpretable stylometric and
  lexical features (length, vocabulary richness, readability, punctuation,
  urgency cues, and 70 function-word frequencies). Standard library + numpy only,
  so every feature is explainable in the viva.
- **Models** (`src/train.py`): logistic-regression baseline, Random Forest and
  XGBoost, each with 5-fold grid search under a fixed seed.
- **Evaluation** (`src/evaluate.py`): precision, recall, F1, accuracy, ROC-AUC,
  percentile bootstrap 95% CIs and confusion-matrix plots.
- **Interpretation** (`src/interpret.py`): impurity and permutation importance.
- **Robustness** (`scripts/supplementary_analysis.py`,
  `scripts/grouped_split_eval.py`): structural-cue ablation, similarity-cluster
  grouped partition, and a 2022-only sensitivity check.
- **Screen audit** (`scripts/audit_screen_sample.py`, `audit_screen_score.py`):
  stratified manual audit of the credential-harvesting screen.
- **Validation** (`scripts/validate_final_alignment.py`): 40 automated checks that
  the dissertation's tables match the freshly generated outputs.

## Prerequisites

- **Python 3.9+**
- **macOS only:** XGBoost links against the OpenMP runtime, which macOS does not
  ship. Without it `import xgboost` fails with
  `Library not loaded: @rpath/libomp.dylib`:

  ```bash
  brew install libomp
  ```

- **Optional**, only for regenerating the dissertation PDF and its contents pages:
  LibreOffice (`brew install --cask libreoffice`) and Poppler
  (`brew install poppler`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Run every script from this directory (`MS-Artifacts/`) with `PYTHONPATH=.`**
> The scripts import `src` and `scripts` as packages, so without it they fail with
> `ModuleNotFoundError: No module named 'src'`. Activating the venv is not enough.

## Getting the data

Raw phishing text is **not** distributed with this repository (see the ethics note
below). Download the two published sources into `data/raw/` yourself:

```bash
mkdir -p data/raw && cd data/raw

# Class 0 — real-world phishing (Nazario corpus, CC BY 4.0)
for y in 2022 2023 2024; do curl -sLO "https://monkey.org/~jose/phishing/phishing-$y"; done

# Class 1 — known AI-generated phishing (Eze & Shamir 2024)
curl -sLO "https://people.cs.ksu.edu/~lshamir/data/ai_phishing/AI_phishing_emails.tar.gz"
tar xzf AI_phishing_emails.tar.gz
cd ../..
```

Expected layout:

```
data/raw/phishing-2022   phishing-2023   phishing-2024
data/raw/AI_phishing_emails/email_1.txt ... email_928.txt   (865 files; indices are not contiguous)
data/raw/AI_phishing_emails.tar.gz
```

`data/corpus_summary.json` records the SHA-256 of each archive, so you can confirm
you obtained identical inputs.

`scripts/build_corpus.py` then produces `data/emails.csv` with **three** columns:

| column | meaning |
|---|---|
| `text` | the email body, plain text |
| `label` | `0` = real-world phishing, `1` = known AI-generated phishing |
| `source` | originating archive — **required** by the 2022-only sensitivity analysis |

> **Ethics — read before you touch data.** This is dual-use security research.
> Do not send any message to any recipient. Do not target real people or
> organisations. Store data on an encrypted device, exclude raw phishing text from
> any public release, and delete it at project end per university policy. No human
> participants are involved, so no information sheet or consent form is required.

## Run the full experiment

Run in this order, from `MS-Artifacts/`, with the venv active:

```bash
# 1. build the corpus from the raw archives
PYTHONPATH=. python scripts/build_corpus.py

# 2. train and evaluate the three classifiers
PYTHONPATH=. python -m src.pipeline --data data/emails.csv --out results/

# 3. class means, feature importance and the structural-cue ablation
PYTHONPATH=. python scripts/supplementary_analysis.py

# 4. similarity-cluster grouped partition and 2022-only sensitivity
PYTHONPATH=. python scripts/grouped_split_eval.py

# 5. credential-screen audit (sample, then score the recorded judgements)
PYTHONPATH=. python scripts/audit_screen_sample.py
PYTHONPATH=. python scripts/audit_screen_score.py

# 6. figures
PYTHONPATH=. python scripts/make_figures.py

# 7. check every reported number against the generated outputs
PYTHONPATH=. python scripts/validate_final_alignment.py \
    ../MS-Final-Dissertation/Dissertation_Final_Validated.docx
```

Steps 2–4 each take a few minutes because of the grid searches and the clustering.

Optional, only if you are rebuilding the document itself:

```bash
PYTHONPATH=. python scripts/make_alignment_matrix.py      # proposal alignment matrix
PYTHONPATH=. python scripts/finalise_validated_docx.py    # insert figures, update Appendix A
PYTHONPATH=. python scripts/repaginate_toc.py             # refresh contents pages + PDF
```

`scripts/build_final_dissertation.py` generates the document from scratch. It is
retained for provenance, but the submitted document is produced from the aligned
baseline by `finalise_validated_docx.py`; running the from-scratch builder will
**not** reproduce the submitted wording.

## Expected results

If every step is followed, you should obtain exactly these values. Any deviation
means something differs in your inputs or environment.

**Corpus:** 982 messages — 491 real-world, 491 known AI-generated.
**Split:** 686 train / 148 validation (reserved, unused) / 148 test (74 per class), seed 42.

| Model | Precision | Recall | F1 | F1 95% CI | ROC-AUC | Accuracy |
|---|---|---|---|---|---|---|
| Logistic regression | 0.974 | 1.000 | 0.987 | [0.965, 1.000] | 0.995 | 0.986 |
| Random Forest | 0.986 | 0.959 | 0.973 | [0.942, 0.994] | 0.999 | 0.973 |
| XGBoost | 0.986 | 0.973 | 0.980 | [0.953, 1.000] | 0.998 | 0.980 |

Logistic-regression confusion matrix: TN 72, FP 2, FN 0, TP 74.

| Robustness condition | Test n | Logistic | RF | XGBoost |
|---|---|---|---|---|
| Structural-cue ablation | 148 | 0.974 | 0.973 | 0.973 |
| Similarity-cluster grouped partition | 149 | 1.000 | 0.993 | 1.000 |
| 2022-only sensitivity (86/class) | 26 | 0.963 | 0.963 | 0.963 |

Step 7 prints `40/40 checks passed` and exits 0 when everything reconciles.

## Verify the machinery without the real corpus

```bash
PYTHONPATH=. python -m pytest -q            # unit tests for features + metrics
PYTHONPATH=. python scripts/make_sample_data.py   # NEUTRAL smoke-test data
PYTHONPATH=. python scripts/smoke_test.py         # end-to-end run on neutral data
```

The neutral sample data exists **only** to prove the code runs end to end. It is
synthetic and must never be reported as a result.

## Reproducibility

Every stochastic step is seeded (seed 42): corpus down-sampling, the stratified
split, the grid searches, each model, the bootstrap, the grouped partition and the
audit sample. Re-running on the same raw archives reproduces `data/emails.csv`
byte-for-byte and every reported number to three decimal places.

## Repository layout

```
src/        features.py  data_loader.py  train.py  evaluate.py  interpret.py
            metrics.py   pipeline.py
tests/      test_features.py
scripts/    build_corpus.py              supplementary_analysis.py
            grouped_split_eval.py        audit_screen_sample.py
            audit_screen_score.py        make_figures.py
            validate_final_alignment.py  make_alignment_matrix.py
            finalise_validated_docx.py   repaginate_toc.py
            build_final_dissertation.py  make_sample_data.py  smoke_test.py
data/       raw/ (you add the archives)  emails.csv  corpus_summary.json
results/    generated outputs (gitignored except the two validation deliverables)
requirements.txt   config.yaml   README.md
```

## Mapping to dissertation chapters

| Chapter | Backed by |
|---|---|
| Design | `src/features.py` feature spec; `results/fig_architecture.png` |
| Implementation | all `src/` modules, `tests/`, `scripts/build_corpus.py`, the screen audit |
| Evaluation | `results/results.json`, `supplementary.json`, `grouped_split.json` |
| Discussion | feature-importance interpretation and limitations |
| Appendix B | `data/corpus_summary.json` funnel and provenance |

---

*Author: Agboola Michael Daramola · Supervisor: Dr Radu Negoescu · University of Lancashire*
