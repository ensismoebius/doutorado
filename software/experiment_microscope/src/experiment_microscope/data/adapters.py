"""``ExperimentAdapter`` ABC + shared return types (FIXME §41).

The GUI does not care whether an artifact came from meeting01 or thesis. Each
adapter maps its pipeline's on-disk layout and its C++ entry points onto one
common vocabulary. Anything an adapter cannot supply is returned as a
``MISSING`` value or an empty list — never faked.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiment_microscope.core.integrity import Origin, Value


@dataclass(frozen=True)
class TreeNode:
    """One row in the Data Explorer (FIXME §7)."""

    kind: str                 # "experiment" | "dataset" | "subject" | "recording" | "sample" | "run" | "fold" | "model" | "result"
    label: str
    handle: Any               # opaque; passed back to the adapter to descend
    metadata: dict[str, Any] = field(default_factory=dict)
    has_children: bool = True


@dataclass(frozen=True)
class Signal1D:
    """A raw or preprocessed 1-D trace (FIXME §8)."""

    samples: np.ndarray       # shape (n,) or (channels, n)
    sample_rate: float
    origin: Origin
    channel_names: tuple[str, ...] = ()
    unit: str = ""
    label: str = ""


@dataclass(frozen=True)
class ProvenanceRecord:
    """FIXME §27 — surfaced verbatim from artifacts wherever possible."""

    source: dict[str, Value]      # experiment / dataset / subject / trial / window
    processing: dict[str, Value]  # wavelet / level / normalization / encoding / time_steps
    model: dict[str, Value]       # architecture / v_th / latent_dim / seed
    artifact: dict[str, Value]    # file / path / git_commit / config_hash / fold / run


@dataclass(frozen=True)
class FeatureMatrix:
    """FIXME §14 — one feature set, samples x features."""

    values: np.ndarray            # (n_samples, n_features)
    feature_names: tuple[str, ...]
    sample_labels: tuple[str, ...]
    class_labels: tuple[Any, ...]
    origin: Origin
    set_label: str = ""


@dataclass(frozen=True)
class ParaconsistentPoint:
    """FIXME §12 — the six quantities the thesis implementation actually
    produces. Field names and formula are fixed by
    ``ThesisParaconsistent.hpp``."""

    label: str
    alpha: Value
    beta: Value
    g1: Value
    g2: Value
    d_truth: Value
    d_penalized: Value
    facet: dict[str, Any] = field(default_factory=dict)  # dataset / modality / wavelet / fold ...


@dataclass(frozen=True)
class LatentTrace:
    """FIXME §19, §20 — one sample through one autoencoder."""

    latent: np.ndarray            # (latent_dim,) or (T, latent_dim)
    reconstruction: np.ndarray    # aligned with the model input
    original: np.ndarray
    spikes: np.ndarray | None     # (T, n_neurons) for SNN, else None
    v_mem: np.ndarray | None
    origin: Origin
    metrics: dict[str, Value] = field(default_factory=dict)  # MSE / MAE / R2 / correlation


class ExperimentAdapter(abc.ABC):
    """Common interface over one research pipeline (FIXME §41)."""

    #: "meeting01" | "thesis" | "paraconsistent_ga"
    key: str = ""
    #: Human label for the explorer root.
    title: str = ""

    # -- hierarchy -------------------------------------------------
    @abc.abstractmethod
    def root_nodes(self) -> list[TreeNode]:
        ...

    @abc.abstractmethod
    def children(self, node: TreeNode) -> list[TreeNode]:
        ...

    # -- per-object payloads (raise BindingUnavailableError when the
    #    payload requires recomputation and nn_microscope is absent) ---
    def load_signal(self, node: TreeNode) -> Signal1D:
        raise NotImplementedError

    def load_provenance(self, node: TreeNode) -> ProvenanceRecord:
        raise NotImplementedError

    def load_features(self, node: TreeNode) -> FeatureMatrix:
        raise NotImplementedError

    def load_wavelet(self, node: TreeNode, **params: Any):
        raise NotImplementedError

    def load_latent(self, node: TreeNode, **params: Any) -> LatentTrace:
        raise NotImplementedError

    def load_metrics(self, node: TreeNode) -> dict[str, Value]:
        raise NotImplementedError

    def paraconsistent_points(self) -> list[ParaconsistentPoint]:
        return []

    def artifact_files(self, node: TreeNode) -> list[str]:
        """On-disk files the selected object originates from (FIXME §28).

        Absolute paths, existing files only. Empty when the object is fully
        recomputed (no persisted artifact)."""
        return []


def missing_map(*names: str) -> dict[str, Value]:
    return {n: Value.missing() for n in names}
