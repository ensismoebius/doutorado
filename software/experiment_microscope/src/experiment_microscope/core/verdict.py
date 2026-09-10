"""Turn a number into a plain sentence a non-specialist can act on.

Explanatory-visualisation research (Knaflic; Segel & Heer) says a title should
carry the *takeaway*, not the category — "the rebuild is near-perfect", not
"residual". These helpers build those sentences from the actual values, and go
through :func:`i18n.t` so the guided tour reads in the chosen language.

They never use words the artifact does not establish ("best", "optimal",
"significant") — they describe magnitude only, so they pass
``integrity.assert_no_inferential_language``.
"""

from __future__ import annotations

import math

from experiment_microscope.core.i18n import t


def _band(x: float, cuts: tuple[float, ...], labels: tuple[str, ...]) -> str:
    for c, lab in zip(cuts, labels):
        if x <= c:
            return lab
    return labels[-1]


def reconstruction(r2: float | None, mse: float | None = None) -> str:
    if r2 is None or math.isnan(r2):
        return t("No rebuild yet — train a model fold first.")
    v = f"{r2:.2f}"
    if r2 >= 0.97:
        return t("Near-perfect rebuild — the small set of latent numbers kept almost everything (R² {r2}).", r2=v)
    if r2 >= 0.85:
        return t("Close rebuild — the shape is right, fine detail is softened (R² {r2}).", r2=v)
    if r2 >= 0.5:
        return t("Rough rebuild — the big movements survive, the sharp parts are lost (R² {r2}).", r2=v)
    if r2 >= 0.0:
        return t("Weak rebuild — close to a flat line at the mean (R² {r2}).", r2=v)
    return t("Failed rebuild — R² {r2} is below zero, so a flat line at the mean would "
             "match the window more closely; the latent numbers did not capture it.", r2=v)


def spikes(n_in: int, n_out: int, n_steps: int) -> str:
    rate = (n_out / n_steps) if n_steps else 0.0
    dens = _band(rate, (0.03, 0.12, 0.30),
                 (t("very sparse"), t("sparse"), t("moderate"), t("dense")))
    kept = "" if n_in == 0 else t(" — the neuron kept {pct}% of the incoming spikes",
                                  pct=f"{100 * n_out / max(n_in, 1):.0f}")
    return t("{n_out} spikes out of {n_steps} time steps ({dens} firing){kept}. "
             "Each spike is one 'the neuron reacted here' moment.",
             n_out=n_out, n_steps=n_steps, dens=dens, kept=kept)


def paraconsistent(alpha: float, beta: float) -> str:
    g1 = alpha - beta
    g2 = alpha + beta - 1.0
    conflict = _band(abs(g2), (0.1, 0.3),
                     (t("clean"), t("somewhat conflicting"), t("highly conflicting")))
    lean = (t("supports the match") if g1 > 0.15 else
            t("denies the match") if g1 < -0.15 else t("is undecided"))
    return t("Evidence {lean} (certainty G1 {g1}) and is {conflict} (contradiction G2 {g2}). "
             "α is support for, β is support against — both can be high at once.",
             lean=lean, g1=f"{g1:+.2f}", conflict=conflict, g2=f"{g2:+.2f}")


def wavelet_energy(rel_energies) -> str:
    import numpy as np

    e = np.asarray(rel_energies, dtype=float)
    if e.size == 0:
        return ""
    top = int(np.argmax(e))
    share = float(e[top] / (e.sum() or 1.0))
    spread = (t("concentrated in a few bands") if share > 0.4
             else t("spread across many bands"))
    return t("The signal's energy is {spread}; band {top} alone holds {pct}%. "
             "Each band is a frequency range, low bands first.",
             spread=spread, top=top, pct=f"{100 * share:.0f}")


def latent(dim: int) -> str:
    return t("Every window — hundreds of samples — is squeezed to just {dim} numbers. "
             "If similar digits land near each other here, those numbers carry the meaning.",
             dim=dim)
