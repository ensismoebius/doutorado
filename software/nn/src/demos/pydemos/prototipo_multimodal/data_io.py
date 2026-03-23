"""Ingestão de dados multimodais (.mat, .npz, wav+csv).

Contrato mínimo retornado por registro:
- audio: np.ndarray shape (n_audio,)
- eeg: np.ndarray shape (channels, n_eeg)
- speaker_id: int
- eeg_index: int (pareamento áudio->EEG quando aplicável)
- sample_id: int
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.io import loadmat
    from scipy.io import wavfile
except ImportError:
    loadmat = None
    wavfile = None


@dataclass(frozen=True)
class RawRecord:
    audio: np.ndarray
    eeg: np.ndarray
    speaker_id: int
    eeg_index: int
    sample_id: int


def _as_1d_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"audio esperado 1D, recebido shape={arr.shape}")
    return arr


def _as_2d_float32_eeg(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"eeg esperado 2D (C,N), recebido shape={arr.shape}")
    return arr


def load_records_from_npz(npz_path: Path) -> list[RawRecord]:
    raise RuntimeError("NPZ record ingestion has been removed from this build; use MAT or wav+csv inputs instead.")


def load_records_from_mat(
    mat_path: Path, audio_var: str = "audio", eeg_var: str = "eeg"
) -> list[RawRecord]:
    if loadmat is None:
        raise ImportError("scipy é necessário para leitura de .mat")
    raw = loadmat(mat_path)
    audio = np.asarray(raw[audio_var], dtype=np.float32)
    eeg = np.asarray(raw[eeg_var], dtype=np.float32)
    speaker_ids = np.asarray(
        raw.get("speaker_id", np.zeros(audio.shape[0])), dtype=np.int64
    ).reshape(-1)
    eeg_index = np.asarray(
        raw.get("eeg_index", np.arange(audio.shape[0])), dtype=np.int64
    ).reshape(-1)
    sample_id = np.asarray(
        raw.get("sample_id", np.arange(audio.shape[0])), dtype=np.int64
    ).reshape(-1)

    records: list[RawRecord] = []
    for i in range(audio.shape[0]):
        records.append(
            RawRecord(
                audio=_as_1d_float32(audio[i]),
                eeg=_as_2d_float32_eeg(eeg[eeg_index[i]]),
                speaker_id=int(speaker_ids[i]),
                eeg_index=int(eeg_index[i]),
                sample_id=int(sample_id[i]),
            )
        )
    return records


def load_records_from_wav_csv(audio_dir: Path, csv_path: Path) -> list[RawRecord]:
    if wavfile is None:
        raise ImportError("scipy é necessário para leitura de wav")
    table = np.genfromtxt(
        csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    eeg_cache: dict[int, np.ndarray] = {}
    records: list[RawRecord] = []
    for row in table:
        wav_name = row["audio_file"]
        speaker_id = int(row["speaker_id"])
        eeg_index = int(row["eeg_index"])
        sample_id = int(row["sample_id"])

        sr, audio = wavfile.read(audio_dir / wav_name)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = _as_1d_float32(audio)

        # Placeholder simples para EEG externo; substitua por um reader real
        if eeg_index not in eeg_cache:
            eeg_cache[eeg_index] = np.zeros((6, 5000), dtype=np.float32)

        records.append(
            RawRecord(
                audio=audio,
                eeg=eeg_cache[eeg_index],
                speaker_id=speaker_id,
                eeg_index=eeg_index,
                sample_id=sample_id,
            )
        )
    return records


def iterate_records(records: Iterable[RawRecord]) -> Iterable[RawRecord]:
    for rec in records:
        yield rec
