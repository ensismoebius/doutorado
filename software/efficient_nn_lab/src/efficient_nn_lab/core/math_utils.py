"""Small numeric helpers shared across bitnet/ and snn/ demos.

Nothing here is specific to quantization or spiking dynamics — just the
generic bits (seeding, interpolation, formatting) that more than one demo
needs, kept in one place instead of copy-pasted.
"""

from __future__ import annotations

import random

import numpy as np

#: Fixed seed used wherever a demo needs *any* randomness (there should be
#: very few such places — ESPECIFICACAO_DLVL.md #35 asks for the main demos
#: to be deterministic without randomness at all).
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation, t in [0, 1]."""
    return a + (b - a) * t


def interpolation_frames(start: float, end: float, n: int) -> list[float]:
    """n evenly spaced values from start to end, inclusive of both ends.

    Used for "sliding" animations (e.g. a weight sliding from its real
    value to its quantized level) so the intermediate frames are exact and
    reproducible rather than driven by wall-clock time.
    """
    if n < 2:
        return [end]
    return [lerp(start, end, t) for t in np.linspace(0.0, 1.0, n)]


def ease_in_out(t: float) -> float:
    """Smoothstep easing (slow-fast-slow), t in [0, 1] -> eased in [0, 1].

    Used for every animated transition in the app: linear motion reads as
    mechanical and makes it hard to tell "still moving" from "arrived";
    easing gives each transition a clear start and a clear settle.
    """
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def tween_values(a: dict[str, object], b: dict[str, object], t: float) -> dict[str, object]:
    """Blend two frame ``values`` dicts at fraction t in [0, 1].

    Numbers, same-length tuples/lists of numbers, and same-shape numpy
    arrays are eased from ``a`` to ``b``. Anything else that can't be
    meaningfully interpolated (labels, stage tags, mismatched shapes)
    simply snaps from ``a`` to ``b`` once t reaches 1 — text should not
    visually "melt" between two unrelated strings.
    """
    eased = ease_in_out(t)
    out: dict[str, object] = {}
    for key in set(a) | set(b):
        if key not in a:
            out[key] = b[key] if t >= 1.0 else None
        elif key not in b:
            out[key] = a[key] if t <= 0.0 else None
        else:
            out[key] = _tween_leaf(a[key], b[key], eased, t)
    return out


def _tween_leaf(va: object, vb: object, eased: float, raw_t: float) -> object:
    if isinstance(va, bool) or isinstance(vb, bool):
        return vb if raw_t >= 1.0 else va
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return lerp(float(va), float(vb), eased)
    if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray) and va.shape == vb.shape:
        return va + (vb - va) * eased
    if isinstance(va, (tuple, list)) and isinstance(vb, (tuple, list)) and len(va) == len(vb):
        blended = [_tween_leaf(x, y, eased, raw_t) for x, y in zip(va, vb)]
        return tuple(blended) if isinstance(va, tuple) else blended
    return vb if raw_t >= 1.0 else va


def fmt(value: object, decimals: int = 3) -> str:
    """Format a number for display, Brazilian decimal comma, fixed width."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.{decimals}f}".replace(".", ",")
    return str(value)
