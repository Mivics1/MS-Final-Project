# MS-Assignments — CO4011 Master's Project

Coursework and dissertation artefact for **CO4011 Master's Project** (MSc
Cybersecurity with AI, University of Lancashire).

**Project:** *Stylometric Detection of LLM-Generated Phishing - Distinguishing 
Known AI-Generated Credential-Harvesting Emails from Real-World Phishing.*

## Contents

| Item | Description |
|---|---|
| `MS-Artifacts/` | The reproducible ML pipeline (see its own README). |
| `MS-Final-Dissertation/` | My project dissertation. |

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
