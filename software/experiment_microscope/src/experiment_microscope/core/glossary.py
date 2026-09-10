"""Plain-language glossary for the whole application (FIXME §29, §33, §55).

The microscope must never show an unexplained abbreviation or a bare number.
Every view pulls its wording from here so the same term is described the same
way everywhere, and the Help → Glossary dialog is generated straight from this
table.

``TERMS[key] = (expansion, meaning)``
  * ``expansion`` — what the letters stand for (``""`` if it is already a word)
  * ``meaning``   — one sentence a non-specialist can act on
"""

from __future__ import annotations

from experiment_microscope.core.i18n import t

TERMS: dict[str, tuple[str, str]] = {
    # -- pipelines / structure ------------------------------------------------
    "meeting01": ("", "The conference-paper pipeline: it trains autoencoders "
                      "(SNN vs LSTM/GRU/Transformer) to reconstruct short audio windows "
                      "and compares them speaker-by-speaker."),
    "thesis": ("", "The doctoral pipeline: handcrafted wavelet features on EEG/voice, "
                   "scored by paraconsistent logic, then used for person authentication."),
    "LOSO": ("Leave-One-Speaker-Out", "A way to test fairly: every speaker is held out "
             "once as the test set while the model trains on all the others, so the score "
             "is 'how well does it work on a person it never saw'."),
    "fold": ("", "One split of the data into train / validation / test. "
                 "Results are averaged over all folds."),
    "outer fold": ("", "The train/test split. 'Outer' because a second, inner split "
                       "(train/validation) sits inside it for model selection."),
    "window": ("", "A short fixed-length slice of the raw signal (here 256 samples). "
                   "The models work on windows, not whole recordings."),
    "AE": ("Autoencoder", "A network that compresses its input to a small vector (the "
           "latent) and then rebuilds it; good reconstruction means the latent kept the "
           "important information."),
    "SNN": ("Spiking Neural Network", "A network whose neurons communicate with discrete "
            "0/1 'spikes' over time instead of continuous numbers — closer to biology and "
            "cheaper on neuromorphic hardware."),
    "LSTM": ("Long Short-Term Memory", "A classic recurrent network for sequences."),
    "GRU": ("Gated Recurrent Unit", "A lighter recurrent network, similar to LSTM."),
    "Transformer": ("", "An attention-based sequence model (the architecture behind modern "
                        "language models)."),
    "PCA": ("Principal Component Analysis", "A rotation of the data onto the directions of "
            "greatest spread; the first few directions give a faithful low-dimensional "
            "picture. A projection, not a measured value."),
    "t-SNE": ("t-distributed Stochastic Neighbour Embedding", "A non-linear 2-D layout that "
              "keeps near points near; good for spotting clusters, but distances and "
              "densities between clusters are not meaningful."),
    "latent": ("", "The small vector an autoencoder's encoder produces (here 32 numbers). "
                   "It is the model's compressed description of the window."),
    "reconstruction": ("", "The decoder's attempt to rebuild the original input from the "
                           "latent vector."),
    "residual": ("", "Original minus reconstruction, point by point. Flat and near zero "
                     "means a faithful rebuild."),

    # -- encodings / SNN internals -----------------------------------------
    "direct": ("", "Spike encoding that passes the normalised sample straight through "
                   "(no conversion to spike times)."),
    "poisson": ("", "Rate encoding: each value becomes a random spike train whose average "
                    "firing rate is proportional to the value."),
    "latency": ("", "Time-to-first-spike encoding: larger values spike earlier."),
    "v_th": ("voltage threshold", "The membrane voltage a LIF neuron must reach to fire a "
             "spike. Higher threshold → fewer spikes."),
    "alpha": ("", "The membrane leak factor (0–1). Closer to 1 = the neuron remembers past "
                  "input longer; closer to 0 = it forgets quickly."),
    "LIF": ("Leaky Integrate-and-Fire", "The standard spiking-neuron model: it adds up "
            "incoming current, leaks some away each step, and fires when it crosses the "
            "threshold, then resets."),
    "membrane potential": ("", "The running internal voltage of a spiking neuron. It rises "
                               "with input, leaks between inputs, and resets after a spike."),
    "v_mem": ("membrane voltage", "Same as membrane potential — the neuron's internal voltage."),
    "time_steps": ("", "How many discrete time steps one sample is unrolled over. "
                       "meeting01's autoencoder uses 1 (the window is fed as a feature "
                       "vector); the recurrent transform uses 256 (the window samples)."),
    "spike raster": ("", "A dot plot: one row per neuron, a mark at every time it fired."),
    "firing rate": ("", "Fraction of time steps on which a neuron spiked (0–1)."),

    # -- wavelet -----------------------------------------------------------
    "wavelet": ("", "A short oscillation used to split a signal into frequency bands at "
                    "several scales at once (unlike a plain spectrum, it keeps time "
                    "information)."),
    "wavelet packet": ("", "A full binary tree of wavelet filters: every band is split "
                           "again, giving equal-width sub-bands (leaves)."),
    "DTWPT": ("Discrete Tunable Wavelet Packet Transform", "The wavelet-packet decomposition "
              "used by the thesis feature extractor."),
    "leaf": ("", "One sub-band at the bottom of the wavelet-packet tree — a narrow "
                 "frequency range."),
    "sub-band": ("", "A limited range of frequencies produced by the wavelet decomposition."),
    "Haar": ("", "The simplest wavelet (a single square step). Fast, blocky."),
    "Daubechies": ("", "A family of smooth wavelets; 'daub10' means 10 filter taps — higher "
                       "number = smoother, wider support."),
    "energy": ("", "Sum of squared coefficients in a band — how much of the signal's power "
                   "sits in that frequency range."),
    "relative energy": ("", "A band's energy divided by the total, so the bands sum to 1."),
    "ZCR": ("Zero-Crossing Rate", "How often the signal changes sign — a rough pitch / "
            "noisiness measure."),
    "entropy": ("", "How evenly spread the coefficients are. High = noise-like, low = a few "
                    "dominant components."),
    "Teager": ("Teager–Kaiser energy operator", "An instantaneous energy estimate sensitive "
               "to both amplitude and frequency."),
    "jitter": ("", "Cycle-to-cycle variation in period — a voice-quality measure."),
    "shimmer": ("", "Cycle-to-cycle variation in amplitude — a voice-quality measure."),
    "LFCC": ("Linear-Frequency Cepstral Coefficients", "A compact spectral-shape descriptor "
             "on a linear frequency axis (the linear-axis cousin of MFCCs)."),
    "cepstral": ("", "Computed from the log spectrum via another transform; captures the "
                     "overall spectral envelope."),

    # -- paraconsistent --------------------------------------------------
    "paraconsistent": ("", "A logic that tolerates contradiction: a statement can be "
                           "supported and denied at the same time. Used here to score how "
                           "cleanly a feature set separates people."),
    "alpha_para": ("evidence FOR (α)", "Degree to which the feature set supports 'same "
                   "person' (0–1)."),
    "beta_para": ("evidence AGAINST (β)", "Degree to which it supports 'different person' "
                  "(0–1)."),
    "G1": ("certainty degree (G1 = α − β)", "How decisive the evidence is. +1 = fully "
           "certain true, −1 = fully certain false, 0 = undecided."),
    "G2": ("contradiction degree (G2 = α + β − 1)", "How much the evidence conflicts with "
           "itself. +1 = fully contradictory, −1 = fully missing, 0 = consistent."),
    "D_truth": ("distance to truth", "Straight-line distance on the (G1, G2) plane from the "
                "point to the ideal 'certainly true, no contradiction' corner (1, 0). "
                "Smaller is better."),
    "D_penalized": ("penalised distance", "D_truth plus a penalty of (2 − √2) × |G2| for "
                    "contradiction. This is the number the feature ranking sorts on — "
                    "smaller is better."),
    "contradiction penalty": ("", "The constant (2 − √2) ≈ 0.5858 that turns contradiction "
                                  "|G2| into extra distance in D_penalized."),

    # -- metrics --------------------------------------------------------
    "MSE": ("Mean Squared Error", "Average of (original − reconstruction)². 0 = perfect; "
            "grows fast with large errors."),
    "MAE": ("Mean Absolute Error", "Average of |original − reconstruction|. 0 = perfect; "
            "in the signal's own units."),
    "R2": ("R² (coefficient of determination)", "Fraction of the signal's variance the "
           "reconstruction captured. 1 = perfect, 0 = no better than a flat line, "
           "negative = worse than flat."),
    "Pearson r": ("Pearson correlation coefficient", "How well the two curves rise and fall "
                  "together, ignoring scale. +1 = identical shape, 0 = unrelated."),
    "pearson_r": ("Pearson correlation coefficient", "How well original and reconstruction "
                  "move together (−1…+1); +1 is identical shape."),
    "EER": ("Equal Error Rate", "The operating point where the chance of wrongly accepting "
            "an impostor equals the chance of wrongly rejecting the genuine user. Lower is "
            "better."),
    "AUC": ("Area Under the ROC Curve", "Probability that a genuine trial scores above an "
            "impostor trial. 1 = perfect, 0.5 = chance."),
    "explained variance": ("", "The share of total spread a PCA component accounts for; the "
                               "first two components' shares tell you how much the 2-D "
                               "picture leaves out."),

    # -- NSGA-II ------------------------------------------------------
    "NSGA-II": ("Non-dominated Sorting Genetic Algorithm II", "A multi-objective search: it "
                "evolves a population of architectures and keeps the ones that are not "
                "beaten on every objective at once."),
    "Pareto front": ("", "The set of solutions where you cannot improve one objective "
                         "without worsening another — the best available trade-offs."),
    "genome": ("", "The encoded description of one candidate architecture the search "
                   "evolves."),
    "feasible": ("", "A candidate that satisfies every hard constraint (e.g. a latency "
                     "budget); infeasible ones are shown but never win."),
    "inference cost": ("", "A hardware-independent proxy for how expensive one forward pass "
                           "is: spike count plus 10 × multiply-accumulates."),
    "UNCALIBRATED": ("", "The estimated latency is a rough model output, not measured on "
                         "real hardware — do not rank or quote it as a real millisecond "
                         "figure."),

    # -- provenance / integrity -----------------------------------------
    "MEASURED": ("", "A value read straight from a saved experiment artifact."),
    "COMPUTED": ("", "A value recomputed here through the exact experiment C++ code — "
                     "deterministic, matches the experiment bit-for-bit."),
    "PROJECTED": ("", "A lossy 2-D/3-D view (PCA, t-SNE) of higher-dimensional data — "
                      "useful for the eye, not a pipeline number."),
    "ESTIMATED": ("", "A value flagged as an approximation (e.g. LIF parameters from a "
                      "checkpoint that predates a fix)."),
    "DISPLAY-DOWNSAMPLED": ("", "The curve you see is decimated for speed; the cursor still "
                                "reads the full-resolution array."),
    "MISSING": ("", "No value is available. Shown as '—', never as 0."),
    "z-score": ("", "The signal after subtracting its mean and dividing by its standard "
                    "deviation, so it is centred on 0 with spread 1 (the models see it "
                    "this way)."),
    "seed": ("", "The random-number-generator starting value. Fixing it makes a run "
                 "reproducible."),
}

# Aliases so a caption can use whichever spelling is natural.
_ALIASES = {
    "r2": "R2", "r²": "R2", "mse": "MSE", "mae": "MAE", "eer": "EER", "auc": "AUC",
    "snn": "SNN", "lstm": "LSTM", "gru": "GRU", "ae": "AE", "pca": "PCA", "lif": "LIF",
    "loso": "LOSO", "lfcc": "LFCC", "zcr": "ZCR", "dtwpt": "DTWPT", "α": "alpha",
    "vth": "v_th", "d_truth": "D_truth", "d_penalized": "D_penalized",
    "alpha": "alpha_para", "beta": "beta_para", "β": "beta_para",
}


def _lookup(term: str) -> tuple[str, str] | None:
    if term in TERMS:
        return TERMS[term]
    low = term.strip().lower()
    if low in _ALIASES:
        return TERMS[_ALIASES[low]]
    for k, v in TERMS.items():
        if k.lower() == low:
            return v
    return None


def expand(term: str) -> str:
    """``'MSE'`` → ``'MSE (Mean Squared Error)'``; unknown term returned as-is."""
    hit = _lookup(term)
    if not hit or not hit[0]:
        return term
    return f"{term} ({t(hit[0])})"


def describe(term: str) -> str:
    """One-sentence meaning, or ``''`` if the term is not in the glossary."""
    hit = _lookup(term)
    return t(hit[1]) if hit else ""


def tooltip(term: str) -> str:
    """Full '<expansion> — <meaning>' string for a widget tooltip."""
    hit = _lookup(term)
    if not hit:
        return ""
    exp, mean = hit
    return f"{t(exp) + ' — ' if exp else ''}{t(mean)}"


def glossary_html() -> str:
    """The whole table as HTML for the Help → Glossary dialog."""
    rows = []
    for term, (exp, mean) in sorted(TERMS.items(), key=lambda kv: kv[0].lower()):
        head = f"<b>{term}</b>" + (f" — <i>{t(exp)}</i>" if exp else "")
        rows.append(f"<p style='margin:4px 0'>{head}<br>{t(mean)}</p>")
    return f"<h3>{t('Glossary')}</h3>" + "".join(rows)
