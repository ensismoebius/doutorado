"""CLI: train the tiny 2-layer spiking vowel classifier on the clips
`record.py` produced, and save the weights `demo.py` loads at runtime.

Run as ``python -m efficient_nn_lab.live.train`` after recording clips for
all five vowels. Prints train/validation accuracy each epoch and writes
``live/weights/vowel_snn_weights.npz``.

Rigor note, since every other number this app shows is either fixed or
disclosed as untrained (see `comparison/autoencoders.py`): the split below
is by CLIP, not by window. Two overlapping windows from the SAME recording
are near-duplicates (pseudoreplication) -- putting one in train and its
neighbor in val would let the network "recognize the recording" rather than
the vowel, and report a validation accuracy that will not hold up live.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from efficient_nn_lab.core.math_utils import SEED
from efficient_nn_lab.live.features import log_energy_bins
from efficient_nn_lab.live.lif_layer import LIFLayerParams, backward_layer, forward_layer
from efficient_nn_lab.live.vowel_model import VOWELS, VowelSnnWeights, encode_poisson, normalize_features, save_weights

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vowel_snn" / "raw"
_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "vowel_snn_weights.npz"


def _load_clips(vowel: str, sample_rate: int) -> list[np.ndarray]:
    clip_dir = _DATA_ROOT / vowel
    clips = sorted(clip_dir.glob("clip_*.npy")) if clip_dir.is_dir() else []
    if not clips:
        raise FileNotFoundError(
            f"no recordings for vowel '{vowel}' in {clip_dir} -- run "
            f"`python -m efficient_nn_lab.live.record --vowel {vowel}` first"
        )
    return [np.load(p) for p in clips]


def _windows_from_clip(clip: np.ndarray, window: int, hop: int) -> list[np.ndarray]:
    return [clip[start : start + window] for start in range(0, max(1, len(clip) - window + 1), hop)]


def _build_dataset(
    sample_rate: int, window_seconds: float, hop_seconds: float, n_bins: int, fmin: float, fmax: float, val_fraction: float, rng: np.random.RandomState
):
    """Returns (train_features, train_labels, val_features, val_labels),
    split by CLIP (see module docstring)."""
    window = int(window_seconds * sample_rate)
    hop = int(hop_seconds * sample_rate)
    train_x, train_y, val_x, val_y = [], [], [], []
    for class_idx, vowel in enumerate(VOWELS):
        clips = _load_clips(vowel, sample_rate)
        order = rng.permutation(len(clips))
        n_val_clips = max(1, round(len(clips) * val_fraction)) if len(clips) > 1 else 0
        val_clip_idx = set(order[:n_val_clips].tolist())
        for i, clip in enumerate(clips):
            for w in _windows_from_clip(clip, window, hop):
                if len(w) < window:
                    continue
                feats = log_energy_bins(w, sample_rate, n_bins, fmin, fmax)
                if i in val_clip_idx:
                    val_x.append(feats)
                    val_y.append(class_idx)
                else:
                    train_x.append(feats)
                    train_y.append(class_idx)
    return (
        np.array(train_x), np.array(train_y),
        np.array(val_x) if val_x else np.zeros((0, n_bins)), np.array(val_y) if val_y else np.zeros(0, dtype=int),
    )


def _init_weights(n_in: int, n_hidden: int, n_out: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    # Small enough that the LIF layers start near-silent rather than
    # saturated -- a freshly initialized network that already fires on
    # every step has no room for training to show any difference at all.
    w1 = rng.normal(0.0, 0.3 / np.sqrt(n_in), size=(n_in, n_hidden))
    w2 = rng.normal(0.0, 0.5 / np.sqrt(n_hidden), size=(n_hidden, n_out))
    return w1, w2


def _forward_backward(features: np.ndarray, target_idx: int, weights: VowelSnnWeights, seed: int):
    intensity = normalize_features(features, weights)
    input_spikes = encode_poisson(intensity, weights, seed=seed)
    hidden = forward_layer(input_spikes, weights.w1, weights.lif)
    output = forward_layer(hidden.spikes, weights.w2, weights.lif)

    t_steps, n_out = output.spikes.shape
    rate = output.spikes.mean(axis=0)
    target = np.zeros(n_out)
    target[target_idx] = 1.0
    grad_rate = 2.0 * (rate - target) / n_out
    grad_output_spikes = np.tile(grad_rate / t_steps, (t_steps, 1))

    grad_w2, grad_hidden_spikes = backward_layer(output, weights.w2, weights.lif, grad_output_spikes)
    grad_w1, _ = backward_layer(hidden, weights.w1, weights.lif, grad_hidden_spikes)
    loss = float(np.mean((rate - target) ** 2))
    predicted = int(np.argmax(rate))
    return loss, predicted, grad_w1, grad_w2


def _evaluate(features: np.ndarray, labels: np.ndarray, weights: VowelSnnWeights, seed: int) -> float:
    if len(features) == 0:
        return float("nan")
    correct = 0
    for feats, label in zip(features, labels):
        intensity = normalize_features(feats, weights)
        input_spikes = encode_poisson(intensity, weights, seed=seed)
        hidden = forward_layer(input_spikes, weights.w1, weights.lif)
        output = forward_layer(hidden.spikes, weights.w2, weights.lif)
        correct += int(np.argmax(output.spikes.mean(axis=0)) == label)
    return correct / len(features)


def train(
    epochs: int = 400,
    lr: float = 0.5,
    lr_decay: float = 0.985,
    hidden: int = 40,
    # 30, not 20: validation confusion measured on real recordings showed
    # /u/ and /o/ -- both back vowels, differing mainly in a lower second
    # formant that a coarser filterbank blurs across fewer, wider bins --
    # as each other's only real mistake. More, narrower bins is the direct
    # lever on that specific confusion (not more epochs, which a fixed
    # bin count cannot fix no matter how long training runs).
    n_bins: int = 30,
    sample_rate: int = 16000,
    window_seconds: float = 0.5,
    hop_seconds: float = 0.1,
    t_steps: int = 16,
    val_fraction: float = 0.25,
    seed: int = SEED,
) -> VowelSnnWeights:
    rng = np.random.RandomState(seed)
    train_x, train_y, val_x, val_y = _build_dataset(
        sample_rate, window_seconds, hop_seconds, n_bins, 80.0, 4000.0, val_fraction, rng
    )
    if len(train_x) == 0:
        raise ValueError("no training windows built -- check data/vowel_snn/raw has recordings for every vowel")

    feature_min = train_x.min(axis=0)
    feature_max = train_x.max(axis=0)
    w1, w2 = _init_weights(n_bins, hidden, len(VOWELS), rng)
    weights = VowelSnnWeights(
        w1=w1, w2=w2, feature_min=feature_min, feature_max=feature_max, n_bins=n_bins,
        sample_rate=sample_rate, window_seconds=window_seconds, t_steps=t_steps,
        lif=LIFLayerParams(),
    )

    # Val accuracy is noisy from one epoch to the next (each window's own
    # Poisson draw, plus a small dataset) -- more epochs alone mostly makes
    # it oscillate longer around the same spot rather than climb, because a
    # FIXED step size keeps overshooting the minimum it is circling. Decay
    # shrinks the step as training goes on, so later epochs actually settle
    # instead of just adding more oscillation; keeping the best-val-seen
    # weights (not whichever epoch happens to be last) is what turns "ran
    # longer" into "more precise" rather than "equally noisy, just slower".
    best_val_acc = -1.0
    best_w1, best_w2 = weights.w1.copy(), weights.w2.copy()
    for epoch in range(epochs):
        current_lr = lr * (lr_decay**epoch)
        order = rng.permutation(len(train_x))
        total_loss, correct = 0.0, 0
        for i in order:
            loss, predicted, grad_w1, grad_w2 = _forward_backward(
                train_x[i], int(train_y[i]), weights, seed=int(rng.randint(0, 2**31 - 1))
            )
            weights.w1 -= current_lr * grad_w1
            weights.w2 -= current_lr * grad_w2
            total_loss += loss
            correct += int(predicted == train_y[i])
        train_acc = correct / len(train_x)
        val_acc = _evaluate(val_x, val_y, weights, seed=seed)
        if len(val_x) and val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_w1, best_w2 = weights.w1.copy(), weights.w2.copy()
        marker = " *" if len(val_x) and val_acc == best_val_acc else ""
        print(
            f"epoch {epoch + 1:3d}/{epochs}  lr={current_lr:.3f}  loss={total_loss / len(train_x):.4f}  "
            f"train_acc={train_acc:.2f}  val_acc={val_acc:.2f}{marker}"
        )

    if len(val_x):
        weights.w1, weights.w2 = best_w1, best_w2
        print(f"melhor val_acc durante o treino: {best_val_acc:.2f} (pesos salvos são desse ponto, não do último epoch)")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.99)
    parser.add_argument("--lr-decay", type=float, default=0.995, help="per-epoch multiplier on --lr")
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--n-bins", type=int, default=30, help="frequency bands in the feature vector")
    parser.add_argument("--out", type=Path, default=_WEIGHTS_PATH)
    args = parser.parse_args()

    weights = train(epochs=args.epochs, lr=args.lr, lr_decay=args.lr_decay, hidden=args.hidden, n_bins=args.n_bins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_weights(str(args.out), weights)
    print(f"pesos salvos em {args.out}")


if __name__ == "__main__":
    main()
