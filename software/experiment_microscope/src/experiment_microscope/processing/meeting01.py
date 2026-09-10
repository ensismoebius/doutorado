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


def recurrent_lif_trace(encoded, alpha: float, v_th: float) -> tuple[np.ndarray, np.ndarray]:
    """``(spikes, v_mem)`` for the recurrent transform — the window samples are the
    time steps, so ``v_mem`` is a real per-step membrane trajectory (FIXME §15/§16)."""
    nm = load_binding()
    out = nm.meeting01.recurrent_lif_trace(np.asarray(encoded, dtype=float), alpha, v_th)
    return np.asarray(out["spikes"]), np.asarray(out["v_mem"])


def flatten(sample) -> np.ndarray:
    nm = load_binding()
    return np.asarray(nm.meeting01.flatten_time_series(np.asarray(sample, dtype=float)))


@dataclass(frozen=True)
class EncoderLayer:
    """One layer's post-forward trace (encoder *or* decoder) (FIXME §15, §18, §19).

    ``kind`` is ``"linear"`` | ``"lif"`` | ``"other"``. ``output`` is that layer's
    activation for the window; ``weight`` is set for linear layers; ``v_mem`` and
    ``voltage_threshold`` for LIF layers (``time_steps == 1`` here, so ``v_mem`` is a
    per-neuron snapshot after the single step, not a trajectory).
    """

    kind: str
    output: np.ndarray | None = None
    weight: np.ndarray | None = None
    v_mem: np.ndarray | None = None
    voltage_threshold: float | None = None


@dataclass(frozen=True)
class AeTrace:
    latent: np.ndarray
    reconstruction: np.ndarray
    encoded_input: np.ndarray
    encoder_layers: tuple[EncoderLayer, ...] = ()
    decoder_layers: tuple[EncoderLayer, ...] = ()
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
    def _layers(key: str) -> tuple[EncoderLayer, ...]:
        built: list[EncoderLayer] = []
        for ld in out.get(key, []) or []:
            d = dict(ld)
            built.append(
                EncoderLayer(
                    kind=str(d.get("type", "other")),
                    output=np.asarray(d["output"]) if "output" in d else None,
                    weight=np.asarray(d["weight"]) if "weight" in d else None,
                    v_mem=np.asarray(d["v_mem"]) if "v_mem" in d else None,
                    voltage_threshold=(
                        float(d["voltage_threshold"]) if "voltage_threshold" in d else None
                    ),
                )
            )
        return tuple(built)

    return AeTrace(
        latent=np.asarray(out["latent"]),
        reconstruction=np.asarray(out["reconstruction"]),
        encoded_input=np.asarray(out["encoded_input"]),
        encoder_layers=_layers("encoder_layers"),
        decoder_layers=_layers("decoder_layers"),
    )
