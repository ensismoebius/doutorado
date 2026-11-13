#!/usr/bin/env python3
"""00_thesis_generate_profiles.py — write every thesis profile (Phase 00 and Phase 01).

The profiles are the experiment's configuration, and the grid is large (275 + 142), so they
are generated here and nowhere else. A hand edit to a generated profile is undone by the
next run, and `profiles.manifest` (read by thesis_profile_audit_gtest) lists exactly what
this script wrote, so a profile added by hand fails the audit.

Signals. The voice, and the EEG under two conditions (EegPairing.hpp):
  voice     the 1974 pronounced voice trials
  eegpron   the EEG of the same trial, recorded while speaking aloud
  eegimag   an Imagined EEG trial of the same speaker and stimulus

Phase 00 (paraconsistent ranking, no classifier), per signal:
  handcrafted  23 wavelets x scales (voice: bark, mel, lfcc; EEG: lfcc) x category
               (c1 band energies, c2 cepstral)
  autoencoder  ANN-AE x 3 sizes + SNN-AE x 3 encodings x 3 sizes
  pca          k = 8, 16, 32
  Every profile also scores the speaker subsets (leave-one-speaker-out and the training
  speakers of every Phase 01 fold: k = 5, seed 42, 3 repetitions).

Phase 01 (authentication). Sources: voice, eegpron, eegimag, and the two fusions under each
EEG condition (earlypron, earlyimag, latepron, lateimag). The extractor is a placeholder
until 02_thesis_apply_winner.py injects the Phase 00 winners.
  p01_dsnn_*    the grid: 7 sources x text (dep, indep) x CV (flat, nested) x (raw, std)
  p01_ann_*     non-spiking twin (the DSNN's graph, ReLU, same init): 7 sources x text,
                flat, std
  p01_cosine_*  no network (cosine on the standardized features): same cells
  p01_dysph_*   simulated dysphonia: sources that read the voice x scope (probes, test) x
                level (0.25, 0.5, 0.75, 1.0); DSNN, dependent, flat, std
  p01_cmp-<family>_*  the three extractor families through the same classifier (Question 3):
                hc / ae / pca x signals (voice, eegpron, eegimag) x text; DSNN, flat, std,
                100 epochs (the autoencoders' Phase 00 budget, so the AE is the one ranked)

BIOMEX-DB replication (confirmatory, decided 2026-11-13; documentation/00-thesis/monography/
issues.md). The Phase 01 cells the thesis concludes on, rerun ONCE on a second base with the
Coretto Phase 00 winners, chosen before any BIOMEX-DB result exists (02_thesis_apply_winner.py
on this directory too). BIOMEX-DB has no imagined speech, so only the pronounced sources:
  bx01_<kind>_*   dsnn / ann / cosine x (voice, eegpron, earlypron, latepron) x text; flat, std
  bx01_dysph_*    the simulated dysphonia of p01_dysph_*, on the sources that read the voice
Nothing is selected on BIOMEX-DB: no Phase 00, no nested CV, no raw (unstandardized) runs.

Run from software/nn:
    python3 scripts/pipeline/thesis/00_thesis_generate_profiles.py
then regenerate the smoke mirror (scripts/testing/thesis_make_smoke_profiles.py).
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src", "experiments",
                                         "thesis", "profiles"))

SEED = 42
REPEATS = 3
FOLD_K = 5
EEG_PAIRING_SEED = 2026  # the imagined-EEG draw; the SAME in both phases
DEGRADATION_SEED = 7

WAVELETS = ["haar"] + [f"daub{n}" for n in range(4, 48, 2)]
DESCRIPTORS = ["energy", "zcr", "entropy", "teager", "jitter", "shimmer"]
SIZES = {"tiny": (16, 8), "small": (32, 16), "base": (64, 32)}
ENCODINGS = ["direct", "latency", "poisson"]
PCA_K = [8, 16, 32]
AE_TRAINING = {"epochs": 100, "learning_rate": 0.001, "samples_per_batch": 32,
               "optimizer_type": "adam", "gradient_clip_norm": 0.0}

SIGNALS = {  # name -> dataset keys
    "voice": {"modality": "voice"},
    "eegpron": {"modality": "eeg", "eeg_condition": "pronounced"},
    "eegimag": {"modality": "eeg", "eeg_condition": "imagined",
                "eeg_pairing_seed": EEG_PAIRING_SEED},
}
SOURCES = dict(SIGNALS)
for mode in ("early", "late"):
    for cond, short in (("pronounced", "pron"), ("imagined", "imag")):
        ds = {"modality": "fused", "fusion_mode": mode, "eeg_condition": cond}
        if cond == "imagined":
            ds["eeg_pairing_seed"] = EEG_PAIRING_SEED
        SOURCES[f"{mode}{short}"] = ds

PLACEHOLDER_EXTRACTOR = {
    "strategy": "handcrafted",
    "handcrafted": {"transform": "dtwpt", "scale": "lfcc", "wavelet": "haar", "cepstral": False,
                    "descriptors": DESCRIPTORS, "dtwpt_level": 4},
}


# Which base each profile reads (ThesisConfig: dataset.source, declared, never inferred).
BASES = {"coretto2017": "~/database.sqlite", "biomex2021": "~/biomex.sqlite"}
BIOMEX_SOURCES = ("voice", "eegpron", "earlypron", "latepron")


def experiment(name):
    return {"run_tag": "e05_" + name, "seed": SEED, "repeats": REPEATS,
            "seed_deterministic": False}


def dataset(source, phase, base="coretto2017"):
    return {"source": base, "root": BASES[base], "results_dir": f"results/thesis/{phase}",
            **SOURCES[source], "max_samples": 0}


# ── Phase 00 ─────────────────────────────────────────────────────────────────────
def handcrafted(wavelet, scale, category):
    cepstral = category == "c2"
    return {"strategy": "handcrafted",
            "handcrafted": {"transform": "dtwpt", "scale": scale, "wavelet": wavelet,
                            "cepstral": cepstral,
                            "descriptors": [d for d in DESCRIPTORS
                                            if not (cepstral and d == "energy")],
                            "dtwpt_level": 4}}


def autoencoder(model, size, encoding=None):
    hidden, latent = SIZES[size]
    ae = {"model": model,
          "encoder_layer_spec": [f"linear:{hidden}:leaky", f"linear:{latent}:identity"],
          "decoder_layer_spec": [f"linear:{hidden}:leaky", "linear:output:identity"]}
    if model == "snn-ae":
        ae["encoding"] = encoding
        if encoding != "direct":
            ae["time_steps"] = 16
        ae["voltage_threshold"] = 1.0 if encoding == "direct" else 0.2
        ae.update({"firing_rate_reg_lambda": 0.5, "firing_rate_min": 0.1,
                   "firing_rate_max": 0.8})
    ae["ae_loss_type"] = "mse"
    return {"strategy": "autoencoder", "autoencoder": ae}


def phase00_extractors(signal):
    scales = ["bark", "mel", "lfcc"] if signal == "voice" else ["lfcc"]
    for w in WAVELETS:
        for s in scales:
            for c in ("c1", "c2"):
                yield f"hc_{w}_{s}_{c}", handcrafted(w, s, c)
    for size in SIZES:
        yield f"ae_ann_{size}", autoencoder("ann-ae", size)
    for enc in ENCODINGS:
        for size in SIZES:
            yield f"ae_snn_{enc}_{size}", autoencoder("snn-ae", size, enc)
    for k in PCA_K:
        yield f"pca_k{k}", {"strategy": "pca", "pca": {"n_components": k}}


def phase00():
    for signal in SIGNALS:
        for ext_name, fe in phase00_extractors(signal):
            name = f"p00_{ext_name}_{signal}"
            prof = {"experiment": experiment(name), "dataset": dataset(signal, "phase00"),
                    "feature_extraction": fe,
                    "paraconsistent": {"enabled": True, "speaker_subsets": True,
                                       "fold_k": FOLD_K, "fold_seed": SEED,
                                       "fold_repeats": REPEATS},
                    "classifier": {"enabled": False}}
            if fe["strategy"] == "autoencoder":
                prof["training"] = dict(AE_TRAINING)
            yield f"phase00/{name}.json", prof


# ── Phase 01 ─────────────────────────────────────────────────────────────────────
def classifier(kind, text):
    c = {"type": kind}
    if kind != "cosine":
        c["layer_spec"] = ["linear:128:relu", "residual:2", "linear:N_speakers:identity"]
    c.update({"text_mode": "dependent" if text == "dep" else "independent", "enabled": True})
    return c


def training(kind, nested, std, epochs=50):
    t = {"k_folds": FOLD_K, "nested_cv": nested, "standardize_features": std}
    if kind == "cosine":
        return t
    t.update({"epochs": epochs, "learning_rate": 0.001, "samples_per_batch": 32,
              "optimizer_type": "adam", "gradient_clip_norm": 0.0, "weight_decay": 0.0})
    if nested:
        t["early_stop_patience"] = 10
    if kind == "dsnn":
        t.update({"firing_rate_reg_lambda": 0.0, "batch_normalization": "none"})
    return t


def phase01_profile(name, source, kind, text, nested, std, fe=None, degradation=None,
                    epochs=50, base="coretto2017", phase="phase01"):
    prof = {"experiment": experiment(name), "dataset": dataset(source, phase, base),
            "feature_extraction": fe or PLACEHOLDER_EXTRACTOR,
            "paraconsistent": {"enabled": False},
            "classifier": classifier(kind, text)}
    if SOURCES[source]["modality"] != "eeg":
        prof["degradation"] = degradation or {"enabled": False}
    prof["training"] = training(kind, nested, std, epochs)
    if (fe or PLACEHOLDER_EXTRACTOR)["strategy"] == "autoencoder":
        # One training block drives both trainings; the AE keeps its Phase 00 budget.
        assert epochs == AE_TRAINING["epochs"]
    return f"{phase}/{name}.json", prof


def phase01():
    for src in SOURCES:
        for text in ("dep", "indep"):
            for cv in ("flat", "nested"):
                for norm in ("raw", "std"):
                    yield phase01_profile(f"p01_dsnn_{src}_{text}_{cv}_{norm}", src, "dsnn",
                                          text, cv == "nested", norm == "std")
            for kind in ("ann", "cosine"):
                yield phase01_profile(f"p01_{kind}_{src}_{text}_flat_std", src, kind, text,
                                      False, True)
    for src in SOURCES:
        if SOURCES[src]["modality"] == "eeg":
            continue  # the degradation touches the voice only
        for scope in ("probes", "test"):
            for level in (0.25, 0.5, 0.75, 1.0):
                deg = {"enabled": True, "level": level, "scope": scope,
                       "seed": DEGRADATION_SEED}
                yield phase01_profile(f"p01_dysph_{src}_{scope}_l{int(level * 100)}", src,
                                      "dsnn", "dep", False, True, degradation=deg)
    families = {"hc": PLACEHOLDER_EXTRACTOR,
                "ae": autoencoder("ann-ae", "base"),  # placeholders, replaced by the winners
                "pca": {"strategy": "pca", "pca": {"n_components": 32}}}
    for fam, fe in families.items():
        for sig in SIGNALS:
            for text in ("dep", "indep"):
                yield phase01_profile(f"p01_cmp-{fam}_{sig}_{text}", sig, "dsnn", text, False,
                                      True, fe=fe, epochs=AE_TRAINING["epochs"])


def biomex():
    """The confirmatory replication on BIOMEX-DB (see the module docstring)."""
    kw = {"base": "biomex2021", "phase": "biomex"}
    for src in BIOMEX_SOURCES:
        assert SOURCES[src].get("eeg_condition", "pronounced") == "pronounced", src
        for text in ("dep", "indep"):
            for kind in ("dsnn", "ann", "cosine"):
                yield phase01_profile(f"bx01_{kind}_{src}_{text}_flat_std", src, kind, text,
                                      False, True, **kw)
    for src in BIOMEX_SOURCES:
        if SOURCES[src]["modality"] == "eeg":
            continue
        for scope in ("probes", "test"):
            for level in (0.25, 0.5, 0.75, 1.0):
                deg = {"enabled": True, "level": level, "scope": scope,
                       "seed": DEGRADATION_SEED}
                yield phase01_profile(f"bx01_dysph_{src}_{scope}_l{int(level * 100)}", src,
                                      "dsnn", "dep", False, True, degradation=deg, **kw)


def main():
    for sub in ("phase00", "phase01", "biomex"):
        shutil.rmtree(os.path.join(PROFILES, sub), ignore_errors=True)
        os.makedirs(os.path.join(PROFILES, sub))
    written = ["debug.json"]  # hand-written, outside the grid
    for rel, prof in list(phase00()) + list(phase01()) + list(biomex()):
        path = os.path.join(PROFILES, rel)
        if os.path.exists(path):
            raise SystemExit(f"[generate] {rel} would be written twice")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2)
            f.write("\n")
        written.append(rel)
    with open(os.path.join(PROFILES, "profiles.manifest"), "w", encoding="utf-8") as f:
        f.write("# Written by scripts/pipeline/thesis/00_thesis_generate_profiles.py; "
                "do not edit.\n")
        for rel in sorted(written):
            f.write(rel + "\n")
    n00 = sum(r.startswith("phase00/") for r in written)
    n01 = sum(r.startswith("phase01/") for r in written)
    nbx = sum(r.startswith("biomex/") for r in written)
    print(f"[generate] {n00} Phase 00 + {n01} Phase 01 + {nbx} BIOMEX-DB profiles -> {PROFILES}")


if __name__ == "__main__":
    main()
