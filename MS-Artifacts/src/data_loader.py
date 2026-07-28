"""
Data loading, cleaning, de-duplication and stratified splitting.

Expected inputs (you supply the real data - see README, Section 'Getting the data'):

  1. A single CSV with columns:  text,label
        label = 0  -> human-written phishing email
        label = 1  -> LLM-generated phishing email
     (Binary task: does writing style alone betray machine authorship?)

  OR

  2. Two folders of raw .txt/.eml files (one per class) passed to
     load_from_folders().

The loader enforces the reproducibility guarantees the methodology promises:
fixed random seed, stratified split, and leakage checks (exact-duplicate removal
BEFORE splitting so no email appears in both train and test).

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _hash(text: str) -> str:
    return hashlib.sha1(_normalise(text).encode("utf-8")).hexdigest()


def clean_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empties and exact duplicates (by normalised hash). Returns a copy."""
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    df["_h"] = df["text"].map(_hash)
    before = len(df)
    df = df.drop_duplicates(subset="_h").drop(columns="_h").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"[data_loader] removed {removed} exact-duplicate emails")
    return df


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns")
    df["label"] = df["label"].astype(int)
    return clean_and_dedupe(df)


def load_from_folders(human_dir: str, llm_dir: str) -> pd.DataFrame:
    rows = []
    for label, folder in [(0, human_dir), (1, llm_dir)]:
        for name in sorted(os.listdir(folder)):
            fp = os.path.join(folder, name)
            if os.path.isfile(fp):
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    rows.append({"text": fh.read(), "label": label})
    return clean_and_dedupe(pd.DataFrame(rows))


@dataclass
class Split:
    X_train_text: List[str]
    X_test_text: List[str]
    y_train: np.ndarray
    y_test: np.ndarray
    X_val_text: List[str]
    y_val: np.ndarray


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Split:
    """Stratified train/val/test split with a fixed seed. Implemented directly so
    the split is reproducible even without scikit-learn."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in sorted(df["label"].unique()):
        idx = df.index[df["label"] == cls].to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

    def take(ix):
        sub = df.loc[ix]
        return sub["text"].tolist(), sub["label"].to_numpy()

    Xtr, ytr = take(train_idx)
    Xva, yva = take(val_idx)
    Xte, yte = take(test_idx)
    print(f"[data_loader] split -> train={len(ytr)} val={len(yva)} test={len(yte)}")
    return Split(Xtr, Xte, ytr, yte, Xva, yva)


def class_balance(y: np.ndarray) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}
