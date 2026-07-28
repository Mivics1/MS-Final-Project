# MS-Assignments — CO4011 Master's Project

Coursework and dissertation artefact for **CO4011 Master's Project** (MSc
Cybersecurity with AI, University of Lancashire).

**Project:** *Stylometric Detection of LLM-Generated Phishing — Distinguishing
AI-Crafted Credential-Harvesting Emails from Human-Written Phishing.*

## Contents

| Item | Description |
|---|---|
| `CO4011_Assessment_Brief.docx` | The module assessment brief (benchmark for marking). |
| `Project_Proposal_DRAFT.docx` | Deliverable 1 — proposal (problem, ethics, Gantt). |
| `Dissertation_Framework.docx` | Chapter framework mapped to the brief & marking matrix. |
| `Unit_1.1_Working_Outline.docx` | Unit 1.1 project-ideas outline. |
| `Unit_1.x` / `Task_1.x` | Unit task worksheets. |
| `MS-Artifacts/` | The reproducible ML pipeline (see its own README). |

## Artefact quick start

```bash
cd MS-Artifacts
pip install -r requirements.txt
python scripts/smoke_test.py         # verify the machinery
python -m src.pipeline --data data/emails.csv --out results/   # real run
```

> **Note:** the phishing corpus is intentionally **not** committed (see
> `.gitignore`). Obtain it from published research sources and place it at
> `MS-Artifacts/data/emails.csv` as described in the artefact README.

---

*Author: Mivics1*
