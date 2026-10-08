"""The trained 2-layer spiking network itself: weights, forward pass, and
load/save -- the one place that knows how a feature vector becomes two
layers of spikes.

Feature normalization here is min-max to [0, 1], fit on the TRAINING
clips only, deliberately not z-score: `poisson_spike_frames` needs an
INTENSITY in [0, 1] (it is a firing *probability*), and a z-scored value
is unbounded in both directions. Feeding a z-score in would silently clip
at `spike_probability`'s own `np.clip(..., 0.0, 1.0)` -- every feature
below the training mean would clip to a flat 0% fire rate, which looks
like "no signal" on screen instead of "below average", a trap worth
naming rather than hitting by surprise during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from efficient_nn_lab.live.lif_layer import LIFLayerParams, forward_layer
from efficient_nn_lab.snn.encoding import poisson_spike_frames

VOWELS: tuple[str, ...] = ("a", "e", "i", "o", "u")


@dataclass
class VowelSnnWeights:
    w1: np.ndarray  # (n_in, n_hidden)
    w2: np.ndarray  # (n_hidden, n_out)
    feature_min: np.ndarray  # (n_in,) -- fit on train clips only
    feature_max: np.ndarray  # (n_in,)
    class_names: tuple[str, ...] = VOWELS
    n_bins: int = 20
    fmin: float = 80.0
    fmax: float = 4000.0
    sample_rate: int = 16000
    window_seconds: float = 0.4
    t_steps: int = 16
    max_rate: float = 0.9
    lif: LIFLayerParams = None  # type: ignore[assignment]  -- set in __post_init__

    def __post_init__(self) -> None:
        if self.lif is None:
            self.lif = LIFLayerParams()


def normalize_features(features: np.ndarray, weights: VowelSnnWeights) -> np.ndarray:
    span = np.maximum(weights.feature_max - weights.feature_min, 1e-9)
    return np.clip((features - weights.feature_min) / span, 0.0, 1.0)


def encode_poisson(intensity: np.ndarray, weights: VowelSnnWeights, seed: int | None) -> np.ndarray:
    """``intensity`` (n_in,) in [0, 1] -> input spikes (T, n_in).

    A fixed ``seed`` makes a run reproducible (training, tests); ``None``
    draws a fresh seed from numpy's own global entropy -- used for live
    inference, where a literally repeating draw would make the network
    react identically to two genuinely different recordings only because
    they hit the same coin flips.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    return poisson_spike_frames(intensity, weights.t_steps, weights.max_rate, seed=seed)


def forward(features: np.ndarray, weights: VowelSnnWeights, seed: int | None = None):
    """features (n_in,) raw (un-normalized) log-energy bins -> (hidden_cache, output_cache).

    Both returned caches are `lif_layer.LayerCache`; inference callers only
    need `.spikes` off each, training needs the full cache for `backward_layer`.
    """
    intensity = normalize_features(features, weights)
    input_spikes = encode_poisson(intensity, weights, seed)
    hidden_cache = forward_layer(input_spikes, weights.w1, weights.lif)
    output_cache = forward_layer(hidden_cache.spikes, weights.w2, weights.lif)
    return hidden_cache, output_cache


def save_weights(path: str, weights: VowelSnnWeights) -> None:
    np.savez(
        path,
        w1=weights.w1,
        w2=weights.w2,
        feature_min=weights.feature_min,
        feature_max=weights.feature_max,
        class_names=np.array(weights.class_names),
        n_bins=weights.n_bins,
        fmin=weights.fmin,
        fmax=weights.fmax,
        sample_rate=weights.sample_rate,
        window_seconds=weights.window_seconds,
        t_steps=weights.t_steps,
        max_rate=weights.max_rate,
        tau=weights.lif.tau,
        v_th=weights.lif.v_th,
        dt=weights.lif.dt,
        surrogate_k=weights.lif.surrogate_k,
    )


def load_weights(path: str) -> VowelSnnWeights:
    data = np.load(path, allow_pickle=False)
    return VowelSnnWeights(
        w1=data["w1"],
        w2=data["w2"],
        feature_min=data["feature_min"],
        feature_max=data["feature_max"],
        class_names=tuple(str(x) for x in data["class_names"]),
        n_bins=int(data["n_bins"]),
        fmin=float(data["fmin"]),
        fmax=float(data["fmax"]),
        sample_rate=int(data["sample_rate"]),
        window_seconds=float(data["window_seconds"]),
        t_steps=int(data["t_steps"]),
        max_rate=float(data["max_rate"]),
        lif=LIFLayerParams(
            tau=float(data["tau"]), v_th=float(data["v_th"]), dt=float(data["dt"]), surrogate_k=float(data["surrogate_k"])
        ),
    )
