"""Unit tests for multimodal_eeg_audio/wavelet_features.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pywt = pytest.importorskip("pywt", reason="PyWavelets not installed")

from wavelet_features import (  # noqa: E402
    _energy,
    _entropy,
    _variance,
    dwt_stats,
    extract_multimodal_wavelet_features,
)


class TestInternalMetrics:
    def test_energy_all_zeros(self):
        assert _energy(np.zeros(10)) == 0.0

    def test_energy_all_ones(self):
        assert abs(_energy(np.ones(8)) - 8.0) < 1e-6

    def test_variance_constant(self):
        assert abs(_variance(np.full(10, 5.0))) < 1e-9

    def test_entropy_all_zeros_is_zero(self):
        assert _entropy(np.zeros(10)) == 0.0

    def test_entropy_nonnegative(self):
        x = np.random.randn(20)
        assert _entropy(x) >= 0.0


class TestDwtStats:
    SIGNAL_LEN = 128
    MAX_LEVEL = 4

    @pytest.fixture()
    def signal(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal(self.SIGNAL_LEN).astype(np.float32)

    def test_output_length(self, signal):
        feats = dwt_stats(signal, family="db4", max_level=self.MAX_LEVEL)
        # wavedec returns (max_level + 1) coefficient arrays; 3 stats each
        expected_len = 3 * (self.MAX_LEVEL + 1)
        assert len(feats) == expected_len

    def test_output_dtype_is_float32(self, signal):
        feats = dwt_stats(signal, family="db4", max_level=self.MAX_LEVEL)
        assert feats.dtype == np.float32

    def test_output_is_finite(self, signal):
        feats = dwt_stats(signal, family="db4", max_level=self.MAX_LEVEL)
        assert np.all(np.isfinite(feats))

    def test_different_signals_differ(self):
        rng = np.random.default_rng(42)
        s1 = rng.standard_normal(128).astype(np.float32)
        s2 = rng.standard_normal(128).astype(np.float32)
        f1 = dwt_stats(s1)
        f2 = dwt_stats(s2)
        assert not np.allclose(f1, f2)


class TestExtractMultimodalWaveletFeatures:
    AUDIO_LEN = 256
    EEG_CHANNELS = 4
    EEG_LEN = 128

    @pytest.fixture()
    def audio(self):
        rng = np.random.default_rng(1)
        return rng.standard_normal(self.AUDIO_LEN).astype(np.float32)

    @pytest.fixture()
    def eeg(self):
        rng = np.random.default_rng(2)
        return rng.standard_normal((self.EEG_CHANNELS, self.EEG_LEN)).astype(np.float32)

    def test_output_is_1d(self, audio, eeg):
        feats = extract_multimodal_wavelet_features(audio, eeg)
        assert feats.ndim == 1

    def test_output_finite(self, audio, eeg):
        feats = extract_multimodal_wavelet_features(audio, eeg)
        assert np.all(np.isfinite(feats))

    def test_output_dtype_float32(self, audio, eeg):
        feats = extract_multimodal_wavelet_features(audio, eeg)
        assert feats.dtype == np.float32

    def test_output_length_matches_channels(self, audio, eeg):
        max_level = 4
        feats = extract_multimodal_wavelet_features(audio, eeg, max_level=max_level)
        stats_per_level = 3
        n_coeff_arrays = max_level + 1
        audio_feats = stats_per_level * n_coeff_arrays
        eeg_feats = self.EEG_CHANNELS * audio_feats
        assert len(feats) == audio_feats + eeg_feats
