"""Turn a number into a plain sentence a non-specialist can act on.

Explanatory-visualisation research (Knaflic; Segel & Heer) says a title should
carry the *takeaway*, not the category — "the rebuild is near-perfect", not
"residual". These helpers build those sentences from the actual values.

They never use words the artifact does not establish ("best", "optimal",
"significant") — they describe magnitude only, so they pass
``integrity.assert_no_inferential_language``.
"""

from __future__ import annotations

import math


def _band(x: float, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for c, lab in zip(cuts, labels):
        if x <= c:
            return lab
    return labels[-1]


def reconstruction(r2: float | None, mse: float | None = None) -> str:
    """One line for the Reconstruction tab title."""
    if r2 is None or math.isnan(r2):
        return "No rebuild yet — train a model fold first."
    if r2 >= 0.97:
        return f"Near-perfect rebuild — the small set of latent numbers kept almost everything (R² {r2:.2f})."
    if r2 >= 0.85:
        return f"Close rebuild — the shape is right, fine detail is softened (R² {r2:.2f})."
    if r2 >= 0.5:
        return f"Rough rebuild — the big movements survive, the sharp parts are lost (R² {r2:.2f})."
    if r2 >= 0.0:
        return f"Weak rebuild — close to a flat line at the mean (R² {r2:.2f})."
    return (f"Failed rebuild — R² {r2:.2f} is below zero, so a flat line at the mean "
            "would match the window more closely; the latent numbers did not capture it.")


def spikes(n_in: int, n_out: int, n_steps: int) -> str:
    """One line for the encoding / SNN spike panels."""
    rate = (n_out / n_steps) if n_steps else 0.0
    dens = _band(rate, (0.03, 0.12, 0.30),
                 ("very sparse", "sparse", "moderate", "dense"))
    passed = "" if n_in == 0 else f" — the neuron kept {100 * n_out / max(n_in, 1):.0f}% of the incoming spikes"
    return (f"{n_out} spikes out of {n_steps} time steps ({dens} firing){passed}. "
            "Each spike is one 'the neuron reacted here' moment.")


def paraconsistent(alpha: float, beta: float) -> str:
    """One line for the paraconsistent plane / triangle / landscape."""
    g1 = alpha - beta
    g2 = alpha + beta - 1.0
    conflict = _band(abs(g2), (0.1, 0.3), ("clean", "somewhat conflicting", "highly conflicting"))
    lean = ("supports the match" if g1 > 0.15 else
            "denies the match" if g1 < -0.15 else "is undecided")
    return (f"Evidence {lean} (certainty G1 {g1:+.2f}) and is {conflict} "
            f"(contradiction G2 {g2:+.2f}). "
            "α is support for, β is support against — both can be high at once.")


def wavelet_energy(rel_energies) -> str:
    """One line for the wavelet sub-band table / plot."""
    import numpy as np

    e = np.asarray(rel_energies, dtype=float)
    if e.size == 0:
        return ""
    top = int(np.argmax(e))
    share = float(e[top] / (e.sum() or 1.0))
    spread = "concentrated in a few bands" if share > 0.4 else "spread across many bands"
    return (f"The signal's energy is {spread}; band {top} alone holds "
            f"{100 * share:.0f}%. Each band is a frequency range, low bands first.")


def latent(dim: int) -> str:
    return (f"Every window — hundreds of samples — is squeezed to just {dim} numbers. "
            "If similar digits land near each other here, those numbers carry the meaning.")
