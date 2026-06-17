"""Unit tests for multimodal_eeg_audio/preprocess.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess import _zscore, _window_1d, _window_eeg  # noqa: E402


class TestZscore:
    def test_zero_mean(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _zscore(x)
        assert abs(float(np.mean(out))) < 1e-6

    def test_unit_std(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = _zscore(x)
        assert abs(float(np.std(out)) - 1.0) < 1e-5

    def test_constant_input_returns_zeros(self):
        x = np.full(10, 3.0)
        out = _zscore(x)
        assert np.allclose(out, 0.0)

    def test_output_shape_preserved(self):
        x = np.random.randn(50)
        assert _zscore(x).shape == (50,)


class TestWindow1d:
    def test_window_count(self):
        n, win, hop = 100, 20, 10
        x = np.arange(n, dtype=np.float32)
        wins = _window_1d(x, win, hop)
        expected = (n - win) // hop + 1
        assert len(wins) == expected

    def test_each_window_correct_size(self):
        x = np.arange(100, dtype=np.float32)
        for w in _window_1d(x, 20, 10):
            assert len(w) == 20

    def test_short_signal_returns_empty(self):
        x = np.arange(5, dtype=np.float32)
        assert _window_1d(x, 10, 5) == []

    def test_first_window_content(self):
        x = np.arange(100, dtype=np.float32)
        wins = _window_1d(x, 10, 5)
        assert np.array_equal(wins[0], np.arange(10, dtype=np.float32))


class TestWindowEeg:
    CHANNELS = 4
    N_SAMPLES = 100
    WIN_SIZE = 20
    HOP = 10

    @pytest.fixture()
    def eeg(self):
        return np.random.randn(self.CHANNELS, self.N_SAMPLES).astype(np.float32)

    def test_window_count(self, eeg):
        wins = _window_eeg(eeg, self.WIN_SIZE, self.HOP)
        expected = (self.N_SAMPLES - self.WIN_SIZE) // self.HOP + 1
        assert len(wins) == expected

    def test_each_window_shape(self, eeg):
        wins = _window_eeg(eeg, self.WIN_SIZE, self.HOP)
        for w in wins:
            assert w.shape == (self.CHANNELS, self.WIN_SIZE)

    def test_short_eeg_returns_empty(self, eeg):
        short = eeg[:, : self.WIN_SIZE - 1]
        assert _window_eeg(short, self.WIN_SIZE, self.HOP) == []
