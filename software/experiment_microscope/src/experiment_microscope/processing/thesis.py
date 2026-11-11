"""Thesis Phase-00 chain recomputation via ``nn_microscope.thesis``.

``load_dataset → extract_handcrafted_features → rank_feature_sets`` — the exact
C++ path. Verified bit-for-bit against a persisted
``results/thesis/phase00/*_paraconsistent.csv`` row (see
``tests/test_thesis_parity.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from experiment_microscope.core.integrity import Origin, Value
from experiment_microscope.processing._binding import load_binding


@dataclass(frozen=True)
class HandcraftedSpec:
    """A handcrafted Phase-00 extractor, every field declared.

    No field has a default: a spec built from a guess (a tag that did not mention the
    scale, a summary that did not record the descriptors) would recompute a different
    extractor than the run used and show it as that run's numbers. ``fusion_mode`` is
    required for ``modality == "fused"`` and must be None otherwise.
    """

    transform: str
    wavelet: str
    scale: str
    cepstral: bool
    modality: str
    dtwpt_level: int
    descriptors: tuple[str, ...]
    fusion_mode: str | None

    def __post_init__(self):
        if (self.modality == "fused") != (self.fusion_mode is not None):
            raise ValueError(
                f"fusion_mode is declared exactly when modality is 'fused' "
                f"(modality={self.modality!r}, fusion_mode={self.fusion_mode!r})")

    @classmethod
    def from_config(cls, config: dict) -> HandcraftedSpec:
        """From a run's declared config: a profile, or a summary's ``config`` block.

        Raises KeyError on any key the run must have declared, and ValueError for a
        non-handcrafted strategy.
        """
        fe = config["feature_extraction"]
        if fe["strategy"] != "handcrafted":
            raise ValueError(f"strategy is {fe['strategy']!r}, not 'handcrafted'")
        hc = fe["handcrafted"]
        ds = config["dataset"]
        modality = ds["modality"]
        return cls(
            transform=hc["transform"],
            wavelet=hc["wavelet"],
            scale=hc["scale"],
            cepstral=bool(hc["cepstral"]),
            modality=modality,
            dtwpt_level=int(hc["dtwpt_level"]),
            descriptors=tuple(hc["descriptors"]),
            fusion_mode=ds["fusion_mode"] if modality == "fused" else None,
        )


def rank_handcrafted(spec: HandcraftedSpec, *, root: str, max_samples: int):
    """Return ``[(label, {alpha..d_penalized: Value})]`` recomputed live."""
    nm = load_binding()
    view = nm.thesis.load_dataset(str(root), max_samples)
    # Handcrafted features are deterministic: the binding takes no seed.
    feature_sets = nm.thesis.extract_handcrafted_features(
        view,
        modality=spec.modality,
        transform=spec.transform,
        scale=spec.scale,
        descriptors=list(spec.descriptors),
        dtwpt_level=spec.dtwpt_level,
        wavelet=spec.wavelet,
        cepstral=spec.cepstral,
        fusion_mode=spec.fusion_mode,
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
