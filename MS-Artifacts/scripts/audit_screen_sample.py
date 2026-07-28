"""
Draw a stratified random sample for MANUAL validation of the
credential-harvesting screen (addresses the reviewer's point that a
deterministic rule can still make systematic errors).

The screen decides which messages enter the corpus, so its validity underpins
the claim that the study is about credential harvesting specifically. This
script re-runs the same parsing/cleaning path as build_corpus.py, records the
screen's decision for every message, and writes a seeded stratified sample of
RETAINED and EXCLUDED messages from BOTH sources to a review file.

The reviewer (project author) then marks each sampled message in
results/screen_audit_labels.json with:
    "y"  - genuinely credential-harvesting
    "n"  - not credential-harvesting
and audit_screen_score.py turns those judgements into precision/recall
estimates for the screen.

Bodies are truncated in the review file (no full phishing text is written to
any published artefact) and the file is gitignored.

Run:  PYTHONPATH=. .venv/bin/python scripts/audit_screen_sample.py
Outputs: results/screen_audit_sample.txt (human-readable, gitignored)
         results/screen_audit_sample.json (ids + decision, gitignored)

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import json
import mailbox
import os
import random

from scripts.build_corpus import (
    AI_DIR, MAX_CHARS, MIN_CHARS, NAZARIO_FILES, RAW,
    body_from_message, is_credential_harvesting, looks_english, normalise,
)

SEED = 42
PER_STRATUM = 25          # 25 x 4 strata = 100 messages audited
SNIPPET = 420             # characters shown per message


def collect():
    """Return (source, id, text, screen_decision) for every cleaned message."""
    out = []
    for fname in NAZARIO_FILES:
        path = os.path.join(RAW, fname)
        for i, msg in enumerate(mailbox.mbox(path)):
            if "FOLDER INTERNAL DATA" in str(msg.get("Subject", "")):
                continue
            text = normalise(body_from_message(msg))
            if not (MIN_CHARS <= len(text) <= MAX_CHARS) or not looks_english(text):
                continue
            out.append(("nazario", f"{fname}#{i}", text, is_credential_harvesting(text)))
    for name in sorted(os.listdir(AI_DIR)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(AI_DIR, name), encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        if lines and lines[0].lower().startswith("header:"):
            lines = lines[1:]
        text = normalise("\n".join(lines))
        if not (MIN_CHARS <= len(text) <= MAX_CHARS):
            continue
        out.append(("ai", name, text, is_credential_harvesting(text)))
    return out


def main() -> None:
    rng = random.Random(SEED)
    rows = collect()
    strata = {
        ("nazario", True): [], ("nazario", False): [],
        ("ai", True): [], ("ai", False): [],
    }
    for src, mid, text, decision in rows:
        strata[(src, decision)].append((mid, text))

    sample, meta = [], []
    for (src, decision), items in strata.items():
        rng.shuffle(items)
        for mid, text in items[:PER_STRATUM]:
            key = f"{src}|{'retained' if decision else 'excluded'}|{mid}"
            sample.append((key, src, decision, text))
            meta.append({"key": key, "source": src, "screen_retained": decision})

    os.makedirs("results", exist_ok=True)
    with open("results/screen_audit_sample.txt", "w", encoding="utf-8") as fh:
        fh.write("CREDENTIAL-HARVESTING SCREEN — MANUAL AUDIT SAMPLE\n")
        fh.write(f"seed={SEED}  per_stratum={PER_STRATUM}  total={len(sample)}\n")
        fh.write("Mark each entry 'y' (is credential harvesting) or 'n' in "
                 "results/screen_audit_labels.json\n\n")
        for key, src, decision, text in sample:
            fh.write("=" * 78 + "\n")
            fh.write(f"KEY: {key}\nSCREEN: {'RETAINED' if decision else 'EXCLUDED'}\n")
            fh.write("-" * 78 + "\n")
            fh.write(text[:SNIPPET].replace("\n", " ") + "\n\n")
    with open("results/screen_audit_sample.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    counts = {f"{s}|{'retained' if d else 'excluded'}": len(v)
              for (s, d), v in strata.items()}
    print("population per stratum:", json.dumps(counts, indent=2))
    print(f"sampled {len(sample)} messages -> results/screen_audit_sample.txt")


if __name__ == "__main__":
    main()
