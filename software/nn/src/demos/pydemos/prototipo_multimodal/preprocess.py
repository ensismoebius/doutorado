"""Pré-processamento: resampling, windowing e concatenação multimodal.

Mantém as regras do PLANNING.md:
- Janela 100 ms
- Áudio em 16 kHz -> 1600 amostras/janela
- EEG em 200 Hz -> 20 amostras/janela/canal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

from .data_io import RawRecord


@dataclass(frozen=True)
class WindowedRecord:
    x_concat: np.ndarray  # (F,)
    audio_win: np.ndarray  # (1600,)
    eeg_win: np.ndarray  # (C,20)
    speaker_id: int
    sample_id: int


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return x - mean
    return (x - mean) / std


def _resample_1d(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal.astype(np.float32, copy=False)
    if resample_poly is None:
        raise ImportError("scipy é necessário para resample")
    # Fração reduzida simples para up/down; estável e reprodutível
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    out = resample_poly(signal, up=up, down=down)
    return np.asarray(out, dtype=np.float32)


def _window_1d(x: np.ndarray, win_size: int, hop: int) -> list[np.ndarray]:
    if x.shape[0] < win_size:
        return []
    return [x[i : i + win_size] for i in range(0, x.shape[0] - win_size + 1, hop)]


def _window_eeg(eeg: np.ndarray, win_size: int, hop: int) -> list[np.ndarray]:
    # eeg shape (C, N)
    c, n = eeg.shape
    if n < win_size:
        return []
    wins: list[np.ndarray] = []
    for i in range(0, n - win_size + 1, hop):
        wins.append(eeg[:, i : i + win_size])
    return wins


def build_windowed_records(
    records: Iterable[RawRecord],
    *,
    audio_orig_sr: int,
    eeg_orig_sr: int,
    target_audio_sr: int = 16_000,
    target_eeg_sr: int = 200,
    window_sec: float = 0.1,
    overlap: float = 0.5,
    zscore_per_window: bool = True,
) -> list[WindowedRecord]:
    win_audio = int(round(target_audio_sr * window_sec))
    win_eeg = int(round(target_eeg_sr * window_sec))
    hop_audio = int(round(win_audio * (1.0 - overlap)))
    hop_eeg = int(round(win_eeg * (1.0 - overlap)))
    hop_audio = max(hop_audio, 1)
    hop_eeg = max(hop_eeg, 1)

    out: list[WindowedRecord] = []
    for rec in records:
        audio_rs = _resample_1d(rec.audio, audio_orig_sr, target_audio_sr)
        eeg_rs = np.vstack(
            [
                _resample_1d(rec.eeg[ch], eeg_orig_sr, target_eeg_sr)
                for ch in range(rec.eeg.shape[0])
            ]
        ).astype(np.float32)

        audio_wins = _window_1d(audio_rs, win_audio, hop_audio)
        eeg_wins = _window_eeg(eeg_rs, win_eeg, hop_eeg)
        n = min(len(audio_wins), len(eeg_wins))

        for i in range(n):
            aw = np.asarray(audio_wins[i], dtype=np.float32)
            ew = np.asarray(eeg_wins[i], dtype=np.float32)
            if zscore_per_window:
                aw = _zscore(aw)
                ew = _zscore(ew)

            x = np.concatenate([aw, ew.reshape(-1)], axis=0).astype(np.float32)
            out.append(
                WindowedRecord(
                    x_concat=x,
                    audio_win=aw,
                    eeg_win=ew,
                    speaker_id=rec.speaker_id,
                    sample_id=rec.sample_id,
                )
            )
    return out
