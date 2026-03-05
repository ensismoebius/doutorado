"""Dataset e split por locutor/sessão para evitar leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import WindowedRecord


@dataclass(frozen=True)
class FeatureSample:
    x: np.ndarray
    y: int
    sample_id: int


class MultimodalWindowDataset(Dataset):
    def __init__(self, records: list[WindowedRecord]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        rec = self._records[idx]
        return {
            "x": torch.from_numpy(rec.x_concat),
            "speaker_id": int(rec.speaker_id),
            "sample_id": int(rec.sample_id),
        }


def split_train_val_by_sample(
    records: list[WindowedRecord], val_ratio: float = 0.2
) -> tuple[list[WindowedRecord], list[WindowedRecord]]:
    sample_ids = np.array([r.sample_id for r in records], dtype=np.int64)
    unique = np.unique(sample_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_set = set(unique[:n_val].tolist())
    train: list[WindowedRecord] = []
    val: list[WindowedRecord] = []
    for r in records:
        if r.sample_id in val_set:
            val.append(r)
        else:
            train.append(r)
    return train, val
