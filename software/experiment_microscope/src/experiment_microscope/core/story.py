"""Guided tours through each pipeline (didactic redesign).

Segel & Heer's "martini glass": a single narrated path first, free exploration
after. Each :class:`Step` is one plain-language sentence or two — no jargon that
has not been introduced — plus the one action that puts the matching view in
front of the audience with real data loaded.

The narration is written for someone who has never seen a neural network. Every
technical term is either avoided or unpacked on the spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Step:
    title: str
    narration: str                         # HTML, plain language
    tab: str = ""                           # tab to bring forward
    select: Callable[["object"], object] | None = None   # workspace -> (node, adapter_key) | None
    after: Callable[["object"], None] | None = None      # extra action once the tab is shown
    callout: Callable[["object"], str] | str = ""        # "the number that matters"


@dataclass
class Story:
    key: str
    title: str
    steps: list[Step] = field(default_factory=list)


# ---- tree helpers -------------------------------------------------------
def _find(adapter, pred, *, root=None, depth=7):
    node = root if root is not None else adapter.root_nodes()[0]
    if pred(node):
        return node
    if depth <= 0:
        return None
    try:
        kids = adapter.children(node)
    except Exception:  # noqa: BLE001
        return None
    for k in kids:
        hit = _find(adapter, pred, root=k, depth=depth - 1)
        if hit is not None:
            return hit
    return None


def _lvl(n):
    return (getattr(n, "handle", {}) or {}).get("level")


def _fsdd_window(ws):
    a = ws.repo.adapter("meeting01")
    n = _find(a, lambda x: _lvl(x) == "window")
    return (n, "meeting01") if n is not None else None


def _thesis_hc_run(ws):
    a = ws.repo.adapter("thesis")
    n = _find(a, lambda x: _lvl(x) == "run" and "hc_" in x.label and "lfcc" in x.label)
    if n is None:
        n = _find(a, lambda x: _lvl(x) == "run")
    return (n, "thesis") if n is not None else None


def _thesis_hc_sample(ws):
    a = ws.repo.adapter("thesis")
    run = _find(a, lambda x: _lvl(x) == "run" and "hc_" in x.label)
    if run is None:
        return None
    n = _find(a, lambda x: _lvl(x) == "sample", root=run, depth=4)
    return (n, "thesis") if n is not None else None


# ---- callouts (computed live from the shown view) ----------------------
def _recon_callout(ws):
    from experiment_microscope.core import verdict

    t = getattr(ws.reconstruction, "_trace", None)
    if t is None:
        return "Train a LOSO fold with save_models: true to see this live."
    r2 = t.metrics.get("r2")
    return verdict.reconstruction(None if r2 is None or r2.is_missing else float(r2.magnitude))


def _latent_callout(ws):
    from experiment_microscope.core import verdict

    return verdict.latent(32)


# ---- the two stories --------------------------------------------------
MEETING01 = Story("meeting01", "How a spiking network learns a spoken digit", [
    Step(
        "1 · A spoken digit is just a wiggle of air",
        "This is the digit spoken aloud, drawn as air pressure over time — the same "
        "thing your ear receives. Press <b>🔊 Listen</b> to hear it.<br><br>"
        "The whole goal: get a computer to tell digits apart <i>without ever being "
        "told</i> what each one sounds like.",
        tab="Signal", select=_fsdd_window,
        callout="256 numbers — one 32-millisecond slice of the sound.",
    ),
    Step(
        "2 · Split the sound into frequency bands",
        "The same slice, broken into 16 <b>frequency bands</b> — low pitches on the "
        "left, high on the right (this split is called a <i>wavelet</i>).<br><br>"
        "Speech energy piles up in just a few bands. That pattern is a fingerprint "
        "of the sound — more useful to a model than the raw wiggle.",
        tab="Wavelet Lab", select=_fsdd_window,
        callout=lambda ws: _wavelet_callout(ws),
    ),
    Step(
        "3 · Turn the numbers into spikes",
        "Real neurons don't pass numbers around — they fire brief <b>spikes</b>. "
        "Here every value in the window is turned into a spike train.<br><br>"
        "With <i>latency</i> encoding, a louder value fires its spike earlier. "
        "Louder = sooner.",
        tab="Encoding Lab", select=_fsdd_window,
        callout=lambda ws: _spikes_callout(ws),
    ),
    Step(
        "4 · One neuron that leaks, charges, and fires",
        "Watch the green line: charge <b>builds up</b> inside the neuron as spikes "
        "arrive, and slowly <b>leaks away</b> between them. When it crosses the red "
        "line the neuron <b>fires</b> a spike and the charge drops.<br><br>"
        "That is a <i>leaky integrate-and-fire</i> neuron — the membrane-potential "
        "picture people ask about.",
        tab="SNN Lab", select=_fsdd_window,
        callout="Green = charge, red = the firing line, amber ticks = spikes out.",
    ),
    Step(
        "5 · Squeeze the whole window to 32 numbers",
        "The full network compresses each window down to just <b>32 numbers</b> — "
        "the <i>latent</i>. Press <b>Project</b>: every window of this fold is run "
        "through and drawn as a dot.<br><br>"
        "If the same digit lands in the same clump, those 32 numbers have captured "
        "what makes the digit that digit.",
        tab="Latent Space", select=_fsdd_window,
        after=lambda ws: _press(ws.latent_explorer, "_go"),
        callout=_latent_callout,
    ),
    Step(
        "6 · Rebuild it — and see what was lost",
        "The decoder tries to redraw the original window from those 32 numbers "
        "alone. <span style='color:#4F9DF7'>Blue</span> = original, "
        "<span style='color:#F5A623'>orange</span> = rebuild, "
        "<span style='color:#969aa0'>grey</span> = the difference.<br><br>"
        "A grey line that barely moves means the 32 numbers kept almost everything.",
        tab="Reconstruction", select=_fsdd_window,
        callout=_recon_callout,
    ),
    Step(
        "You've seen the whole path",
        "Sound → frequency bands → spikes → one neuron's charge → 32 numbers → "
        "rebuild. <br><br>Now click <b>Free explore</b>: pick any window, any tab, "
        "and follow a single number all the way through. Every plot shows where "
        "its data came from.",
    ),
])

THESIS = Story("thesis", "How wavelets + paraconsistent logic authenticate a person", [
    Step(
        "1 · Brain or voice signals from one person",
        "Six channels of <b>EEG</b> (tiny voltages from the scalp) or a voice "
        "recording. The goal: decide whether two recordings come from the "
        "<i>same person</i>.",
        tab="Signal", select=_thesis_hc_sample,
        callout="EEG here is 6 channels × 4096 samples at 1024 readings per second.",
    ),
    Step(
        "2 · Split into frequency bands",
        "The same wavelet idea as before: break each channel into frequency "
        "bands. The hand-designed path then measures several things in every band "
        "— energy, how often it crosses zero, how disordered it is, and more.",
        tab="Wavelet Lab", select=_thesis_hc_sample,
        callout=lambda ws: _wavelet_callout(ws),
    ),
    Step(
        "3 · 96 hand-picked measurements per recording",
        "Each row is one recording, each column one measurement. Unlike the "
        "spoken-digit network, nothing here is learned — a human chose every "
        "measurement. This is the <i>handcrafted feature matrix</i>.",
        tab="Feature Matrix", select=_thesis_hc_run,
        callout="1974 recordings × 96 measurements for this run.",
    ),
    Step(
        "4 · Evidence can support AND deny at once",
        "<b>Paraconsistent</b> logic allows a claim to be backed and contradicted "
        "at the same time — exactly what noisy biometrics look like.<br><br>"
        "Horizontal <b>G1</b> = net certainty (right = 'same person'). Vertical "
        "<b>G2</b> = how much the evidence fights itself. The ideal spot is the "
        "far right at zero height.",
        tab="Paraconsistent plane", select=_thesis_hc_run,
        callout=lambda ws: _para_callout(ws),
    ),
    Step(
        "5 · Which recipe of measurements wins",
        "Every combination of wavelet + measurements gets one score: its distance "
        "to that ideal spot, with a penalty for self-contradiction "
        "(<i>D_penalized</i> — smaller is better).<br><br>"
        "This sorted table is the experiment's actual answer.",
        tab="Ranking", select=_thesis_hc_run,
        callout="The row at the top has the least contradiction and the most certainty.",
    ),
    Step(
        "You've seen the whole path",
        "Signal → frequency bands → 96 measurements → support-vs-denial plane → "
        "ranking. <br><br>Click <b>Free explore</b> and follow any feature set "
        "through the Triangle tab to see all three stages side by side.",
    ),
])

STORIES = {s.key: s for s in (MEETING01, THESIS)}


# ---- small helpers used by callouts / after -------------------------
def _press(widget, attr):
    btn = getattr(widget, attr, None)
    if btn is not None:
        try:
            btn.click()
        except Exception:  # noqa: BLE001
            pass


def _wavelet_callout(ws):
    from experiment_microscope.core import verdict

    d = getattr(ws.wavelet_lab, "_decomp", None)
    if d is None:
        return ""
    return verdict.wavelet_energy(d.subband_energies)


def _spikes_callout(ws):
    from experiment_microscope.core import verdict
    import numpy as np

    el = ws.encoding_lab
    w = getattr(el, "_window", None)
    if w is None:
        return ""
    try:
        from experiment_microscope.processing import meeting01 as m

        d = np.asarray(m.encode(w.reshape(-1, 1), "latency", 0)).reshape(-1)
        n_out = int((d > 0.5).sum())
        return verdict.spikes(0, n_out, w.size)
    except Exception:  # noqa: BLE001
        return ""


def _para_callout(ws):
    from experiment_microscope.core import verdict

    pts = getattr(ws.para_plane, "_points", None)
    if not pts:
        return ""
    p = pts[0]
    if p.alpha.is_missing or p.beta.is_missing:
        return ""
    return verdict.paraconsistent(float(p.alpha.magnitude), float(p.beta.magnitude))
