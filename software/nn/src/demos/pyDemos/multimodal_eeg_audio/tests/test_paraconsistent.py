"""Unit tests for multimodal_eeg_audio/paraconsistent.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paraconsistent import (  # noqa: E402
    certainty_and_contradiction,
    mu_lambda_from_probabilities,
    summarize_paraconsistent,
)


class TestMuLambdaFromProbabilities:
    def _uniform_probs(self, n: int, n_classes: int) -> np.ndarray:
        return np.full((n, n_classes), 1.0 / n_classes, dtype=np.float32)

    def test_mu_is_correct_class_probability(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
        y = np.array([0, 1])
        mu, _ = mu_lambda_from_probabilities(probs, y)
        assert abs(mu[0] - 0.7) < 1e-6
        assert abs(mu[1] - 0.8) < 1e-6

    def test_lambda_is_max_wrong_class(self):
        probs = np.array([[0.6, 0.3, 0.1]], dtype=np.float32)
        y = np.array([0])
        _, lam = mu_lambda_from_probabilities(probs, y)
        assert abs(lam[0] - 0.3) < 1e-6

    def test_shapes_match_n_samples(self):
        n, c = 10, 5
        probs = np.random.dirichlet(np.ones(c), size=n).astype(np.float32)
        y = np.random.randint(0, c, size=n)
        mu, lam = mu_lambda_from_probabilities(probs, y)
        assert mu.shape == (n,)
        assert lam.shape == (n,)

    def test_output_dtype_float32(self):
        probs = np.array([[0.5, 0.5]], dtype=np.float64)
        y = np.array([0])
        mu, lam = mu_lambda_from_probabilities(probs, y)
        assert mu.dtype == np.float32
        assert lam.dtype == np.float32


class TestCertaintyAndContradiction:
    def test_gc_formula(self):
        mu = np.array([0.8, 0.6], dtype=np.float32)
        lam = np.array([0.1, 0.3], dtype=np.float32)
        gc, _ = certainty_and_contradiction(mu, lam)
        np.testing.assert_allclose(gc, mu - lam, atol=1e-6)

    def test_gct_formula(self):
        mu = np.array([0.8], dtype=np.float32)
        lam = np.array([0.1], dtype=np.float32)
        _, gct = certainty_and_contradiction(mu, lam)
        expected = mu + lam - 1.0
        np.testing.assert_allclose(gct, expected, atol=1e-6)

    def test_gc_in_minus_one_to_one(self):
        rng = np.random.default_rng(0)
        mu = rng.uniform(0, 1, 50).astype(np.float32)
        lam = rng.uniform(0, 1 - mu, 50).astype(np.float32)
        gc, _ = certainty_and_contradiction(mu, lam)
        assert np.all(gc >= -1.0) and np.all(gc <= 1.0)

    def test_dtype_float32(self):
        mu = np.array([0.7], dtype=np.float32)
        lam = np.array([0.2], dtype=np.float32)
        gc, gct = certainty_and_contradiction(mu, lam)
        assert gc.dtype == np.float32
        assert gct.dtype == np.float32


class TestSummarizeParaconsistent:
    def test_returns_dict_with_keys(self):
        gc = np.array([0.4, 0.6, 0.2], dtype=np.float32)
        gct = np.array([0.1, -0.2, 0.05], dtype=np.float32)
        result = summarize_paraconsistent(gc, gct)
        assert set(result.keys()) == {"gc_mean", "gc_std", "gct_mean", "gct_std"}

    def test_gc_mean_is_correct(self):
        gc = np.array([1.0, 3.0], dtype=np.float32)
        gct = np.zeros(2, dtype=np.float32)
        result = summarize_paraconsistent(gc, gct)
        assert abs(result["gc_mean"] - 2.0) < 1e-5

    def test_all_values_are_floats(self):
        gc = np.random.randn(20).astype(np.float32)
        gct = np.random.randn(20).astype(np.float32)
        result = summarize_paraconsistent(gc, gct)
        for v in result.values():
            assert isinstance(v, float)
