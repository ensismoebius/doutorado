"""Wavelet recomputation via ``nn_microscope.wavelet`` (FIXME §10, §11).

Thin pass-through to the C++ ``wavelets::malat``. No decomposition logic lives
here — that would be the "second, subtly different implementation" FIXME §3
forbids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiment_microscope.core.integrity import Origin
from experiment_microscope.processing._binding import load_binding


@dataclass(frozen=True)
class WaveletDecomposition:
    transformed_signal: np.ndarray
    packet: bool
    levels: int
    leaf_count: int
    subband_energies: np.ndarray
    origin: Origin = Origin.COMPUTED
    wavelet: str = ""
    mode: str = ""

    def leaf(self, index: int) -> np.ndarray:
        """Coefficients of one packet leaf (only meaningful for packet mode)."""
        if not self.leaf_count:
            raise RuntimeError("leaf() is only defined for a packet decomposition")
        n = self.transformed_signal.size // self.leaf_count
        return self.transformed_signal[index * n : (index + 1) * n]


def decompose(signal, wavelet: str = "haar", mode: str = "packet", level: int = 4) -> WaveletDecomposition:
    nm = load_binding()
    sig = [float(x) for x in np.asarray(signal).ravel()]
    result = nm.wavelet.decompose(sig, wavelet, mode, level)
    energies = nm.wavelet.subband_energies(result, level)
    is_packet = bool(result.packet)
    return WaveletDecomposition(
        transformed_signal=np.asarray(result.transformed_signal, dtype=float),
        packet=is_packet,
        levels=int(result.levels),
        leaf_count=int(result.packet_leaf_count()) if is_packet else 0,
        subband_energies=np.asarray(energies, dtype=float),
        wavelet=wavelet,
        mode=mode,
    )
