"""Smoke test for the training loop's math (live/train.py's `_forward_backward`
and `_init_weights`), on tiny synthetic feature clusters -- never real
audio, so this stays deterministic and runs offline in CI. Establishes the
"train for a few steps, assert the loss actually decreases" pattern this
project's test suite did not have before (see live/train.py's own
docstring on why a by-clip split still matters for the real thing).
"""

from __future__ import annotations

import numpy as np

from efficient_nn_lab.live.train import _forward_backward, _init_weights
from efficient_nn_lab.live.vowel_model import LIFLayerParams, VowelSnnWeights

_N_BINS = 6
_N_HIDDEN = 10
_N_CLASSES = 5


def _synthetic_dataset(rng: np.random.RandomState, n_per_class: int = 8):
    """One well-separated synthetic cluster per class: class ``c``'s
    features are centered on a one-hot-ish bump at bin ``c``, so a network
    with enough capacity should be able to tell them apart easily."""
    features, labels = [], []
    for c in range(_N_CLASSES):
        center = np.zeros(_N_BINS)
        center[c % _N_BINS] = 3.0
        for _ in range(n_per_class):
            features.append(np.clip(center + rng.normal(0.0, 0.2, size=_N_BINS), 0.0, None))
            labels.append(c)
    return np.array(features), np.array(labels)


def test_training_loop_reduces_loss_on_separable_synthetic_data():
    rng = np.random.RandomState(0)
    features, labels = _synthetic_dataset(rng)
    w1, w2 = _init_weights(_N_BINS, _N_HIDDEN, _N_CLASSES, rng)
    weights = VowelSnnWeights(
        w1=w1, w2=w2,
        feature_min=features.min(axis=0), feature_max=features.max(axis=0),
        n_bins=_N_BINS, sample_rate=16000, window_seconds=0.1, t_steps=12,
        lif=LIFLayerParams(),
    )

    def epoch_loss() -> float:
        total = 0.0
        for feats, label in zip(features, labels):
            loss, _, _, _ = _forward_backward(feats, int(label), weights, seed=int(rng.randint(0, 2**31 - 1)))
            total += loss
        return total / len(features)

    first_epoch_loss = epoch_loss()
    lr = 0.5
    for _ in range(25):
        order = rng.permutation(len(features))
        for i in order:
            _, _, grad_w1, grad_w2 = _forward_backward(features[i], int(labels[i]), weights, seed=int(rng.randint(0, 2**31 - 1)))
            weights.w1 -= lr * grad_w1
            weights.w2 -= lr * grad_w2
    last_epoch_loss = epoch_loss()

    assert last_epoch_loss < first_epoch_loss * 0.7
