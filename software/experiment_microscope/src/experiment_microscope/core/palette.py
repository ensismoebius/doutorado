"""One colour language for the whole app (didactic redesign).

Research on teaching visualisations (CNN Explainer, TensorFlow Playground) is
unanimous on one point: pick a small set of colours, give each a *meaning*, use
it everywhere, and keep a legend on screen. A blue line means the same thing on
every tab.

Colours are chosen to stay distinguishable for the common colour-vision
deficiencies (blue / orange / grey is the classic safe triad); every place that
uses colour to carry meaning also varies line style or marker so colour is
never the only cue (FIXME §34).
"""

from __future__ import annotations

#: name -> (r, g, b). The name is the *concept*, not the hue.
SEMANTIC: dict[str, tuple[int, int, int]] = {
    "input":        (79, 157, 247),   # the original / encoder input / measured signal
    "output":       (245, 166, 35),   # the rebuild / decoder output / prediction
    "error":        (150, 154, 160),  # residual / difference / "how wrong"
    "spike":        (255, 193, 7),    # a spike event
    "membrane":     (67, 197, 158),   # membrane potential v[t]
    "threshold":    (232, 97, 90),    # firing threshold v_th (also "danger / conflict")
    "wavelet":      (176, 124, 232),  # wavelet sub-band energy
    "feature":      (46, 197, 197),   # handcrafted feature value
    "truth":        (67, 197, 158),   # paraconsistent Truth vertex / clean evidence
    "contradiction": (232, 97, 90),   # paraconsistent G2 / conflicting evidence
    "highlight":    (255, 225, 77),   # the thing you clicked / the cross-link
    "grid":         (110, 110, 110),
}

#: short label for the on-screen legend strip.
MEANING: dict[str, str] = {
    "input": "original signal",
    "output": "model's rebuild",
    "error": "difference (how wrong)",
    "spike": "a spike fired",
    "membrane": "charge in the neuron",
    "threshold": "firing line",
    "wavelet": "frequency-band energy",
    "feature": "one measurement",
    "truth": "clean support",
    "contradiction": "support + denial at once",
    "highlight": "your selection",
}

#: the full sentence, shown as a tooltip on the legend swatch.
MEANING_LONG: dict[str, str] = {
    "input": "the original signal — what goes into the model",
    "output": "the rebuilt signal — what the model produced from the latent numbers",
    "error": "the difference between original and rebuild — how wrong the rebuild is",
    "spike": "a spike — the instant a neuron fired",
    "membrane": "membrane charge building up inside one neuron and leaking between spikes",
    "threshold": "the firing line (v_th) — when the charge crosses it the neuron spikes",
    "wavelet": "energy contained in one frequency band of the signal",
    "feature": "one hand-designed measurement of the signal",
    "truth": "evidence that cleanly supports the answer (near the Truth vertex)",
    "contradiction": "evidence that supports and denies the answer at the same time",
    "highlight": "the item you selected — drawn in this colour on every tab",
}

#: which legend entries matter on which tab (keeps the strip short).
FOR_VIEW: dict[str, tuple[str, ...]] = {
    "Signal": ("input",),
    "Wavelet Lab": ("input", "wavelet"),
    "Wavelet 3D": ("wavelet",),
    "Encoding Lab": ("input", "spike"),
    "SNN Lab": ("input", "spike", "membrane", "threshold"),
    "Latent Space": ("highlight",),
    "Reconstruction": ("input", "output", "error"),
    "Feature Matrix": ("feature",),
    "Voice Through the Network": ("input", "spike", "membrane", "output"),
    "Triangle": ("wavelet", "feature", "truth", "contradiction"),
    "Paraconsistent plane": ("truth", "contradiction"),
    "Paraconsistent landscape": ("truth", "contradiction"),
}


def rgb(name: str) -> tuple[int, int, int]:
    return SEMANTIC.get(name, (200, 200, 200))


def hex_(name: str) -> str:
    r, g, b = rgb(name)
    return f"#{r:02x}{g:02x}{b:02x}"


def pen(name: str, width: int = 1, style: str | None = None):
    """A pyqtgraph pen in the semantic colour. ``style`` in {"dash", "dot"}."""
    from experiment_microscope.views._pg import pg
    from PySide6.QtCore import Qt

    kw = {"color": rgb(name), "width": width}
    if style == "dash":
        kw["style"] = Qt.PenStyle.DashLine
    elif style == "dot":
        kw["style"] = Qt.PenStyle.DotLine
    return pg.mkPen(**kw)


def brush(name: str, alpha: int = 255):
    from experiment_microscope.views._pg import pg

    r, g, b = rgb(name)
    return pg.mkBrush(r, g, b, alpha)


def legend_strip(view_name: str):
    """A one-row widget: swatch + plain meaning for each colour used on this
    tab. Returns a ``QWidget`` (empty and hidden if the tab has no entry)."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

    from experiment_microscope.core.i18n import t

    w = QWidget()
    w.setMaximumHeight(24)
    row = QHBoxLayout(w)
    row.setContentsMargins(8, 1, 8, 1)
    row.setSpacing(16)
    names = FOR_VIEW.get(view_name, ())
    if not names:
        w.setVisible(False)
        return w
    key = QLabel(f"<b>{t('Colour key:')}</b>")
    key.setStyleSheet("color:#999;")
    row.addWidget(key)
    for n in names:
        lab = QLabel(
            f"<span style='color:{hex_(n)};font-size:14px'>■</span> "
            f"<span style='color:#bbb'>{t(MEANING.get(n, n))}</span>"
        )
        lab.setTextFormat(Qt.TextFormat.RichText)
        lab.setToolTip(t(MEANING_LONG.get(n, "")))
        row.addWidget(lab)
    row.addStretch(1)
    return w
