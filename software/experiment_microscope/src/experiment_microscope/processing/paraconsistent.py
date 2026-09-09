"""Paraconsistent scoring via ``nn_microscope.thesis`` (FIXME §12, §55).

Wraps ``thesis::score_feature_set``. The six quantities and the
``d_penalized = d_truth + (2 - sqrt(2)) * |g2|`` formula are the C++
implementation's — this module only marshals arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.processing._binding import load_binding


@dataclass(frozen=True)
class Score:
    label: str
    alpha: float
    beta: float
    g1: float
    g2: float
    d_truth: float
    d_penalized: float

    def as_values(self) -> dict[str, Value]:
        return {
            name: Value(getattr(self, name), Origin.COMPUTED)
            for name in ("alpha", "beta", "g1", "g2", "d_truth", "d_penalized")
        }


def score(feature_vectors, subject_ids, label: str = "adhoc") -> Score:
    nm = load_binding()
    s = nm.thesis.paraconsistent_score(
        [list(map(float, v)) for v in feature_vectors],
        [int(i) for i in subject_ids],
        label,
    )
    return Score(s.label, s.alpha, s.beta, s.g1, s.g2, s.d_truth, s.d_penalized)


def extract_handcrafted(signal, sample_rate: float, **cfg):
    """thesis::extract_handcrafted — one signal -> one handcrafted feature vector."""
    nm = load_binding()
    return list(nm.thesis.extract_handcrafted([float(x) for x in signal], float(sample_rate), **cfg))
