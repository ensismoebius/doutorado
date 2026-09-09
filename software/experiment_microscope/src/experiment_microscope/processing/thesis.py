"""Thesis Phase-00 chain recomputation via ``nn_microscope.thesis``.

``load_dataset → extract_handcrafted_features → rank_feature_sets`` — the exact
C++ path. Verified bit-for-bit against a persisted
``results/thesis/phase00/*_paraconsistent.csv`` row (see
``tests/test_thesis_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.paths import THESIS_DEFAULT_DB
from experiment_microscope.processing._binding import load_binding

# Handcrafted config parsed out of a phase00 run_tag stem, e.g.
#   hc_daub10_lfcc_c1_eeg  ->  wavelet=daub10 scale=lfcc cepstral=False modality=eeg
_SCALES = ("lfcc", "bark", "mel")


@dataclass(frozen=True)
class HandcraftedSpec:
    wavelet: str = "daub4"
    scale: str = "lfcc"
    cepstral: bool = False
    modality: str = "eeg"
    dtwpt_level: int = 4
    descriptors: tuple[str, ...] = ("energy", "zcr", "entropy", "teager", "jitter", "shimmer")

    @classmethod
    def from_run_tag(cls, run_tag: str) -> "HandcraftedSpec":
        parts = run_tag.split("_")
        wavelet = next((p for p in parts if p.startswith("daub") or p == "haar"), "daub4")
        scale = next((p for p in parts if p in _SCALES), "lfcc")
        cepstral = "c2" in parts
        modality = next((p for p in parts if p in ("eeg", "voice", "fused")), "eeg")
        return cls(wavelet=wavelet, scale=scale, cepstral=cepstral, modality=modality)


def rank_handcrafted(spec: HandcraftedSpec, *, root: str | None = None, seed: int = 42):
    """Return ``[(label, {alpha..d_penalized: Value})]`` recomputed live."""
    nm = load_binding()
    view = nm.thesis.load_dataset(str(root or THESIS_DEFAULT_DB), spec.modality, 0)
    feature_sets = nm.thesis.extract_handcrafted_features(
        view,
        modality=spec.modality,
        transform="dtwpt",
        scale=spec.scale,
        descriptors=list(spec.descriptors),
        dtwpt_level=spec.dtwpt_level,
        wavelet=spec.wavelet,
        cepstral=spec.cepstral,
        seed=seed,
    )
    out = []
    for s in nm.thesis.rank_feature_sets(view, feature_sets):
        out.append(
            (
                s.label,
                {
                    name: Value(getattr(s, name), Origin.COMPUTED)
                    for name in ("alpha", "beta", "g1", "g2", "d_truth", "d_penalized")
                },
            )
        )
    return out
