"""Extração de features wavelet para áudio e EEG por janela."""

from __future__ import annotations

import numpy as np

try:
    import pywt
except ImportError:
    pywt = None


def _energy(x: np.ndarray) -> float:
    return float(np.sum(np.square(x)))


def _variance(x: np.ndarray) -> float:
    return float(np.var(x))


def _entropy(x: np.ndarray) -> float:
    p = np.square(x)
    s = float(np.sum(p))
    if s <= 1e-12:
        return 0.0
    p = p / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def dwt_stats(
    signal_1d: np.ndarray, family: str = "db4", max_level: int = 4
) -> np.ndarray:
    if pywt is None:
        raise ImportError(
            "PyWavelets não instalado. Instale com: pip install PyWavelets"
        )
    coeffs = pywt.wavedec(signal_1d, wavelet=family, level=max_level)
    feats: list[float] = []
    for c in coeffs:
        feats.extend([_energy(c), _variance(c), _entropy(c)])
    return np.asarray(feats, dtype=np.float32)


def extract_multimodal_wavelet_features(
    audio_win: np.ndarray, eeg_win: np.ndarray, family: str = "db4", max_level: int = 4
) -> np.ndarray:
    # áudio 1D
    a_feats = dwt_stats(audio_win, family=family, max_level=max_level)
    # EEG por canal
    ch_feats = [
        dwt_stats(eeg_win[ch], family=family, max_level=max_level)
        for ch in range(eeg_win.shape[0])
    ]
    e_feats = np.concatenate(ch_feats, axis=0).astype(np.float32)
    return np.concatenate([a_feats, e_feats], axis=0).astype(np.float32)
