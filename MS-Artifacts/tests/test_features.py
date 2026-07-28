"""
Unit tests for the feature extractor. Run with:  python -m pytest -q
(These use only numpy/stdlib so they run in any environment.)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import features
from src.metrics import binary_metrics, confusion_matrix


def test_feature_vector_length_is_stable():
    a = features.extract_features("Hello world. This is a test!")
    b = features.extract_features("Completely different text, longer and with URLs http://x.co")
    assert list(a.keys()) == list(b.keys()) == features.FEATURE_NAMES


def test_empty_text_is_safe():
    f = features.extract_features("")
    assert f["word_count"] == 0.0
    assert all(np.isfinite(v) for v in f.values())


def test_url_and_email_counts():
    t = "Please verify at http://bad.example/login or email a@b.com now"
    f = features.extract_features(t)
    assert f["url_count"] == 1.0
    assert f["email_addr_count"] >= 1.0
    assert f["urgency_word_ratio"] > 0.0


def test_matrix_shape():
    texts = ["one two three", "four five six seven", ""]
    X = features.extract_matrix(texts)
    assert X.shape == (3, len(features.FEATURE_NAMES))
    assert np.all(np.isfinite(X))


def test_type_token_ratio_bounds():
    f = features.extract_features("the the the the")
    assert 0.0 < f["type_token_ratio"] <= 1.0


def test_confusion_matrix_and_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.tolist() == [[1, 1], [0, 2]]
    m = binary_metrics(y_true, y_pred)
    assert abs(m["recall"] - 1.0) < 1e-9
    assert abs(m["precision"] - (2 / 3)) < 1e-9
