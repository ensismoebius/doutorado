"""meeting01 window-pipeline recomputation via ``nn_microscope.meeting01``.

Thin pass-through to the C++ implementation (FIXME §3, §198). The recompute
chain per window is::

    encode_sample -> apply_snn_architecture_transform -> flatten_time_series -> AE

which is exactly what ``Meeting01Training.cpp`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_microscope.core.integrity import Origin
from experiment_microscope.processing._binding import load_binding


def encode(sample, encoding: str, seed: int = 0) -> np.ndarray:
    nm = load_binding()
    return np.asarray(nm.meeting01.encode_sample(np.asarray(sample, dtype=float), encoding, seed))


def architecture_transform(encoded, architecture: str, alpha: float, v_th: float) -> np.ndarray:
    nm = load_binding()
    return np.asarray(
        nm.meeting01.apply_snn_architecture_transform(
            np.asarray(encoded, dtype=float), architecture, alpha, v_th
        )
    )


def flatten(sample) -> np.ndarray:
    nm = load_binding()
    return np.asarray(nm.meeting01.flatten_time_series(np.asarray(sample, dtype=float)))


@dataclass(frozen=True)
class AeTrace:
    latent: np.ndarray
    reconstruction: np.ndarray
    encoded_input: np.ndarray
    origin: Origin = Origin.COMPUTED


def snn_ae_forward(
    config_path: str,
    *,
    alpha: float,
    v_th: float,
    architecture: str,
    encoder_npz: str,
    decoder_npz: str,
    flat_window,
    encoding: str = "direct",
    seed: int = 0,
) -> AeTrace:
    nm = load_binding()
    out = nm.meeting01.snn_ae_forward(
        config_path, alpha, v_th, architecture, encoder_npz, decoder_npz,
        np.asarray(flat_window, dtype=float), encoding, seed,
    )
    return AeTrace(
        latent=np.asarray(out["latent"]),
        reconstruction=np.asarray(out["reconstruction"]),
        encoded_input=np.asarray(out["encoded_input"]),
    )
