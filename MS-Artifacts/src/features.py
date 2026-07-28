"""
Stylometric and lexical feature extraction for phishing-email authorship detection.

This module converts an email body (plain text) into a fixed-length vector of
interpretable stylometric and lexical features. It deliberately uses only the
Python standard library and numpy so that the SAME extraction code runs both in
a full ML environment (with scikit-learn / xgboost) and in a minimal environment
for testing.

Design rationale (see dissertation, Chapter 4 - Design):
Each feature is chosen because the literature on machine-generated-text detection
reports that LLM output tends to be more fluent, more regular, and less error-prone
than human writing. The features below operationalise those observations so that a
classifier can be trained on them and, crucially, so that feature importance can be
INTERPRETED afterwards (Objective O7).

Author: Agboola Michael Daramola
Module: CO4011 Master's Project
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Lexical resources
# ---------------------------------------------------------------------------

# A compact English function-word list. Function words (determiners, pronouns,
# prepositions, conjunctions, auxiliaries) are the classic stylometric signal:
# they are used unconsciously and their relative frequencies are hard to fake,
# which is why authorship-attribution research has relied on them since Mosteller
# and Wallace (1964).
FUNCTION_WORDS: List[str] = [
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "at", "by", "for", "with", "about", "as", "into", "than", "that", "this",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "may", "might", "must",
    "not", "no", "so", "very", "just", "please", "kindly",
]

# Words frequently associated with urgency / call-to-action in phishing. Included
# as lexical (content) features; useful for interpretation, not as a rule.
URGENCY_WORDS: List[str] = [
    "urgent", "immediately", "verify", "suspended", "account", "password",
    "click", "confirm", "update", "security", "alert", "expire", "expired",
    "limited", "action", "required", "now", "important", "warning", "unusual",
]

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_RE.split(text) if s.strip()]
    return parts


def _count_syllables(word: str) -> int:
    """Approximate syllable count via vowel-group counting (good enough for
    readability indices; exactness is not required)."""
    groups = _VOWEL_RE.findall(word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def extract_features(text: str) -> Dict[str, float]:
    """Return an ordered dict of interpretable features for a single email body."""
    text = text or ""
    words = _tokenize_words(text)
    sentences = _split_sentences(text)
    n_words = len(words)
    n_sents = len(sentences)
    n_chars = len(text)
    unique_words = set(words)

    word_lengths = [len(w) for w in words]
    sent_lengths = [len(_tokenize_words(s)) for s in sentences]

    counts = Counter(words)
    hapax = sum(1 for w, c in counts.items() if c == 1)
    dis = sum(1 for w, c in counts.items() if c == 2)

    # Readability (Flesch Reading Ease and Flesch-Kincaid Grade Level)
    syllables = sum(_count_syllables(w) for w in words)
    asl = _safe_div(n_words, n_sents)          # avg sentence length (words)
    asw = _safe_div(syllables, n_words)        # avg syllables per word
    flesch = 206.835 - 1.015 * asl - 84.6 * asw if n_words else 0.0
    fk_grade = 0.39 * asl + 11.8 * asw - 15.59 if n_words else 0.0

    punctuation = Counter(ch for ch in text if ch in ",.;:!?'\"-()")
    uppercase = sum(1 for ch in text if ch.isupper())
    digits = sum(1 for ch in text if ch.isdigit())

    feats: Dict[str, float] = {}

    # --- Volume / length ---
    feats["char_count"] = float(n_chars)
    feats["word_count"] = float(n_words)
    feats["sentence_count"] = float(n_sents)
    feats["avg_word_length"] = float(np.mean(word_lengths)) if word_lengths else 0.0
    feats["std_word_length"] = float(np.std(word_lengths)) if word_lengths else 0.0
    feats["avg_sentence_length"] = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    feats["std_sentence_length"] = float(np.std(sent_lengths)) if sent_lengths else 0.0

    # --- Vocabulary richness ---
    feats["type_token_ratio"] = _safe_div(len(unique_words), n_words)
    feats["hapax_ratio"] = _safe_div(hapax, n_words)
    feats["dis_legomena_ratio"] = _safe_div(dis, n_words)
    # Honore's R and Yule's K: classic vocabulary-richness statistics
    feats["honore_r"] = (
        100 * math.log(n_words) / (1 - _safe_div(hapax, len(unique_words)))
        if n_words > 1 and len(unique_words) and hapax != len(unique_words)
        else 0.0
    )
    m1 = len(unique_words)
    m2 = sum(c * c for c in counts.values())
    feats["yule_k"] = 1e4 * _safe_div((m2 - m1), (n_words * n_words)) if n_words else 0.0

    # --- Readability ---
    feats["flesch_reading_ease"] = float(flesch)
    feats["flesch_kincaid_grade"] = float(fk_grade)
    feats["avg_syllables_per_word"] = float(asw)

    # --- Punctuation & orthography ---
    feats["comma_ratio"] = _safe_div(punctuation[","], n_chars)
    feats["period_ratio"] = _safe_div(punctuation["."], n_chars)
    feats["exclamation_ratio"] = _safe_div(punctuation["!"], n_chars)
    feats["question_ratio"] = _safe_div(punctuation["?"], n_chars)
    feats["punctuation_ratio"] = _safe_div(sum(punctuation.values()), n_chars)
    feats["uppercase_ratio"] = _safe_div(uppercase, n_chars)
    feats["digit_ratio"] = _safe_div(digits, n_chars)

    # --- Structural / phishing-relevant lexical cues ---
    feats["url_count"] = float(len(_URL_RE.findall(text)))
    feats["email_addr_count"] = float(len(_EMAIL_RE.findall(text)))
    feats["urgency_word_ratio"] = _safe_div(
        sum(counts[w] for w in URGENCY_WORDS), n_words
    )

    # --- Function-word frequencies (stylometric fingerprint) ---
    for fw in FUNCTION_WORDS:
        feats[f"fw_{fw}"] = _safe_div(counts[fw], n_words)

    return feats


# Stable, ordered feature-name list (function words appended in fixed order).
FEATURE_NAMES: List[str] = list(extract_features("the quick brown fox jumps.").keys())


def extract_matrix(texts: List[str]) -> "np.ndarray":
    """Vectorise a list of texts into an (n_samples, n_features) numpy array,
    guaranteeing consistent column order via FEATURE_NAMES."""
    rows = []
    for t in texts:
        f = extract_features(t)
        rows.append([f.get(name, 0.0) for name in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


@dataclass
class FeatureSet:
    """Container pairing the feature matrix with its column names and labels."""
    X: "np.ndarray"
    y: "np.ndarray"
    feature_names: List[str]


def build_feature_set(texts: List[str], labels: List[int]) -> FeatureSet:
    X = extract_matrix(texts)
    y = np.asarray(labels, dtype=int)
    return FeatureSet(X=X, y=y, feature_names=FEATURE_NAMES)
