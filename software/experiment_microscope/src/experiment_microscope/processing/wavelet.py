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


@dataclass(frozen=True)
class LevelBand:
    """One rung of the regular-transform decomposition ladder — either the
    final approximation or one level's detail band. ``level`` names which
    resolution it came from (COMPUTED, real coefficient counts — halving each
    level, never resampled to a common length)."""

    label: str
    level: int
    coefficients: np.ndarray
    origin: Origin = Origin.COMPUTED


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


def decompose_levels(signal, wavelet: str = "haar", level: int = 4) -> list[LevelBand]:
    """The classic multiresolution "ladder": the final approximation plus each
    level's detail band, coarsest to finest — level N detail, level N-1 detail,
    ..., level 1 detail, with the final level-N approximation first.

    This needs the REGULAR (non-packet) transform: only there does
    ``get_wavelet_transforms(detail_index)`` mean "the k-th detail band"
    (0 = approximation, 1..levels = detail k) — packet mode chunks the signal
    into same-sized leaves instead, which is what the 2D Wavelet Lab's existing
    leaf table already shows. This is a second, complementary view, not a
    replacement.
    """
    nm = load_binding()
    sig = [float(x) for x in np.asarray(signal).ravel()]
    result = nm.wavelet.decompose(sig, wavelet, "regular", level)
    levels = int(result.levels)
    bands = [LevelBand(
        label="approximation",
        level=levels,
        coefficients=np.asarray(result.wavelet_transforms(0), dtype=float),
    )]
    for k in range(levels, 0, -1):
        bands.append(LevelBand(
            label=f"detail {k}",
            level=k,
            coefficients=np.asarray(result.wavelet_transforms(k), dtype=float),
        ))
    return bands
