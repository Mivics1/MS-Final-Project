"""
Generate a NEUTRAL synthetic dataset for SMOKE-TESTING THE MACHINERY ONLY.

IMPORTANT - READ THIS:
    This script does NOT create phishing content and its output must NEVER be used
    as results in the dissertation. Its sole purpose is to prove that the feature
    extraction and pipeline glue execute end-to-end and produce sensible numbers.

    The two classes are neutral texts with deliberately different stylometric
    profiles (Class 0 = short, casual, high-variance sentences; Class 1 = longer,
    uniform, formal sentences), so a classifier should separate them - which
    verifies the plumbing, nothing more.

    Your REAL experiment uses a genuine corpus of human-written vs LLM-generated
    phishing emails that you obtain from public research sources (see README).

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

CASUAL_FRAGMENTS = [
    "hey", "yeah so", "idk", "lol", "kinda", "gonna grab food", "brb",
    "that was wild", "no way", "u there", "cool cool", "ok sure", "meh",
    "big mood", "same tbh", "wanna hang", "later maybe",
]

FORMAL_SENTENCES = [
    "The committee reviewed the quarterly performance indicators in detail.",
    "Subsequent analysis confirmed the reliability of the measurement apparatus.",
    "The organisation intends to consolidate its operational procedures accordingly.",
    "A comprehensive assessment of the available evidence was duly undertaken.",
    "The proposed framework aligns with established methodological conventions.",
    "Stakeholders were informed of the revised timeline in a timely manner.",
    "The findings substantiate the hypothesis advanced in the preceding section.",
    "Appropriate controls were implemented to preserve the integrity of the data.",
]


def make_casual(rng: random.Random) -> str:
    n = rng.randint(3, 8)
    parts = [rng.choice(CASUAL_FRAGMENTS) for _ in range(n)]
    # short, punchy, irregular
    return " ".join(parts) + rng.choice(["!", "!!", " ...", "?", ""])


def make_formal(rng: random.Random) -> str:
    n = rng.randint(3, 6)
    return " ".join(rng.choice(FORMAL_SENTENCES) for _ in range(n))


def build(n_per_class: int, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": make_casual(rng), "label": 0})
        rows.append({"text": make_formal(rng), "label": 1})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="samples per class")
    ap.add_argument("--out", default="data/sample_neutral.csv")
    args = ap.parse_args()
    df = build(args.n)
    df.to_csv(args.out, index=False)
    print(f"[make_sample_data] wrote {len(df)} rows to {args.out} "
          f"(NEUTRAL smoke-test data - not for dissertation results)")


if __name__ == "__main__":
    main()
