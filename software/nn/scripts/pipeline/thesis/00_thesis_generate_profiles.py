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
EEG condition (earlypron, earlyimag, latepron, lateimag). The extractor is the Phase 00
winner, read from winners.json (01_thesis_phase00_rank.py) and written here:
    profile                    winner
    p01_cmp-<family>_<signal>  winners[signal].by_family[family]
    voice / eegpron / eegimag  winners[that signal]
    a fusion, condition c      --fused voice -> winners["voice"]; --fused eeg -> winners[eeg of c]
A fused run extracts both signals through one config, so one winner serves both halves; --fused
states which (no default: it is a design choice; the thesis uses voice). There is no
placeholder: before the ranking, --phase00-only writes Phase 00 alone and removes the Phase 01
and BIOMEX-DB profiles, because a profile with an extractor nobody chose would run and give a
plausible number. (Until 2026-11-13 the profiles shipped a placeholder that
02_thesis_apply_winner.py replaced; regenerating without re-applying silently put the PCA
comparison back at k = 32 while the winner is k = 8.)
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
Coretto Phase 00 winners, chosen before any BIOMEX-DB result exists (the same winners.json and
--fused). BIOMEX-DB has no imagined speech, so only the pronounced sources:
  bx01_<kind>_*   dsnn / ann / cosine x (voice, eegpron, earlypron, latepron) x text; flat, std
  bx01_dysph_*    the simulated dysphonia of p01_dysph_*, on the sources that read the voice
Nothing is selected on BIOMEX-DB: no Phase 00, no nested CV, no raw (unstandardized) runs.

Run from software/nn:
    python3 scripts/pipeline/thesis/00_thesis_generate_profiles.py \
        --winners results/thesis/phase00/winners.json --fused voice
    python3 scripts/pipeline/thesis/00_thesis_generate_profiles.py --phase00-only   # before ranking
then regenerate the smoke mirror (scripts/testing/thesis_make_smoke_profiles.py).
profiles.manifest records the winners file's sha256 and --fused; test_thesis_phase_scripts.py
regenerates the tree from them and refuses any difference from the files on disk.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
NN = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
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

FAMILIES = ("hc", "ae", "pca")  # winners.json by_family keys (01_thesis_phase00_rank.py)
EEG_SIGNAL = {"pronounced": "eegpron", "imagined": "eegimag"}


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


def load_winners(path):
    """winners.json, refused unless it names a winner for every signal and family."""
    with open(path, encoding="utf-8") as f:
        winners = json.load(f)
    for sig in SIGNALS:
        w = winners.get(sig)
        if not isinstance(w, dict) or "feature_extraction" not in w or "by_family" not in w:
            raise SystemExit(f"[generate] {path} has no winner for '{sig}' (or no by_family). "
                             f"Re-run 01_thesis_phase00_rank.py after every Phase 00 profile "
                             f"has results; nothing was written.")
        missing = [fam for fam in FAMILIES if fam not in w["by_family"]]
        if missing:
            raise SystemExit(f"[generate] {path}: '{sig}' has no best {missing} extractor; "
                             f"nothing was written.")
    return winners


def winner_for(winners, fused, source, family=None):
    """The Phase 00 winner a Phase 01 profile reads (the table in the module docstring)."""
    if family is not None:
        return winners[source]["by_family"][family]  # source is a signal here
    ds = SOURCES[source]
    if ds["modality"] != "fused":
        return winners[source]
    if fused not in ("voice", "eeg"):
        raise SystemExit(f"[generate] --fused must be voice or eeg, not {fused!r}")
    return winners["voice" if fused == "voice" else EEG_SIGNAL[ds["eeg_condition"]]]


def phase01_profile(name, source, kind, text, nested, std, winner, degradation=None,
                    epochs=50, base="coretto2017", phase="phase01"):
    fe = copy.deepcopy(winner["feature_extraction"])
    if fe["strategy"] == "autoencoder" and (kind == "cosine" or epochs != AE_TRAINING["epochs"]):
        # One training block drives both trainings; running the AE under another budget would
        # evaluate an autoencoder Phase 00 never ranked.
        raise SystemExit(f"[generate] {name}: its winner {winner['label']} is an autoencoder "
                         f"ranked at {AE_TRAINING['epochs']} epochs, but this profile trains "
                         f"{'no network' if kind == 'cosine' else f'{epochs} epochs'}; "
                         f"nothing was written.")
    prof = {"experiment": experiment(name), "dataset": dataset(source, phase, base),
            "feature_extraction": fe,
            "paraconsistent": {"enabled": False},
            "classifier": classifier(kind, text)}
    if SOURCES[source]["modality"] != "eeg":
        prof["degradation"] = degradation or {"enabled": False}
    prof["training"] = training(kind, nested, std, epochs)
    return f"{phase}/{name}.json", prof


def phase01(winners, fused):
    def win(src):
        return winner_for(winners, fused, src)

    for src in SOURCES:
        for text in ("dep", "indep"):
            for cv in ("flat", "nested"):
                for norm in ("raw", "std"):
                    yield phase01_profile(f"p01_dsnn_{src}_{text}_{cv}_{norm}", src, "dsnn",
                                          text, cv == "nested", norm == "std", win(src))
            for kind in ("ann", "cosine"):
                yield phase01_profile(f"p01_{kind}_{src}_{text}_flat_std", src, kind, text,
                                      False, True, win(src))
    for src in SOURCES:
        if SOURCES[src]["modality"] == "eeg":
            continue  # the degradation touches the voice only
        for scope in ("probes", "test"):
            for level in (0.25, 0.5, 0.75, 1.0):
                deg = {"enabled": True, "level": level, "scope": scope,
                       "seed": DEGRADATION_SEED}
                yield phase01_profile(f"p01_dysph_{src}_{scope}_l{int(level * 100)}", src,
                                      "dsnn", "dep", False, True, win(src), degradation=deg)
    for fam in FAMILIES:
        for sig in SIGNALS:
            for text in ("dep", "indep"):
                yield phase01_profile(f"p01_cmp-{fam}_{sig}_{text}", sig, "dsnn", text, False,
                                      True, winner_for(winners, fused, sig, fam),
                                      epochs=AE_TRAINING["epochs"])


def biomex(winners, fused):
    """The confirmatory replication on BIOMEX-DB (see the module docstring)."""
    kw = {"base": "biomex2021", "phase": "biomex"}

    def win(src):
        return winner_for(winners, fused, src)

    for src in BIOMEX_SOURCES:
        assert SOURCES[src].get("eeg_condition", "pronounced") == "pronounced", src
        for text in ("dep", "indep"):
            for kind in ("dsnn", "ann", "cosine"):
                yield phase01_profile(f"bx01_{kind}_{src}_{text}_flat_std", src, kind, text,
                                      False, True, win(src), **kw)
    for src in BIOMEX_SOURCES:
        if SOURCES[src]["modality"] == "eeg":
            continue
        for scope in ("probes", "test"):
            for level in (0.25, 0.5, 0.75, 1.0):
                deg = {"enabled": True, "level": level, "scope": scope,
                       "seed": DEGRADATION_SEED}
                yield phase01_profile(f"bx01_dysph_{src}_{scope}_l{int(level * 100)}", src,
                                      "dsnn", "dep", False, True, win(src), degradation=deg,
                                      **kw)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def manifest_winners(path=None):
    """(winners path relative to software/nn, sha256, fused) as profiles.manifest records them,
    or None for a --phase00-only tree."""
    path = path or os.path.join(PROFILES, "profiles.manifest")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# winners: "):
                rel, _, sha, fused = line[len("# winners: "):].split()
                return rel, sha, fused
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--winners", help="winners.json from 01_thesis_phase00_rank.py")
    what.add_argument("--phase00-only", action="store_true",
                      help="Before the ranking: write Phase 00 and remove the Phase 01 and "
                           "BIOMEX-DB profiles.")
    ap.add_argument("--fused", choices=("voice", "eeg"),
                    help="Which signal's winner the fused sources use (required with --winners).")
    args = ap.parse_args(argv)
    if args.winners and not args.fused:
        ap.error("--winners needs --fused voice|eeg (no default: it is a design choice)")
    if args.phase00_only and args.fused:
        ap.error("--fused applies only with --winners")

    # Build everything before touching the tree: a refusal leaves the profiles as they were.
    rows = list(phase00())
    header = ["# Written by scripts/pipeline/thesis/00_thesis_generate_profiles.py; do not edit."]
    if args.winners:
        winners = load_winners(args.winners)
        rows += list(phase01(winners, args.fused)) + list(biomex(winners, args.fused))
        rel = os.path.relpath(os.path.abspath(args.winners), NN)
        header.append(f"# winners: {rel} sha256 {sha256(args.winners)} {args.fused}")
    else:
        header.append("# phase00 only: no Phase 01 or BIOMEX-DB profile before the winners")
    seen = set()
    for rel, _ in rows:
        if rel in seen:
            raise SystemExit(f"[generate] {rel} would be written twice")
        seen.add(rel)

    for sub in ("phase00", "phase01", "biomex"):
        shutil.rmtree(os.path.join(PROFILES, sub), ignore_errors=True)
    for sub in sorted({rel.split("/")[0] for rel, _ in rows}):
        os.makedirs(os.path.join(PROFILES, sub))
    written = ["debug.json"]  # hand-written, outside the grid
    for rel, prof in rows:
        with open(os.path.join(PROFILES, rel), "w", encoding="utf-8") as f:
            json.dump(prof, f, indent=2)
            f.write("\n")
        written.append(rel)
    with open(os.path.join(PROFILES, "profiles.manifest"), "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for rel in sorted(written):
            f.write(rel + "\n")
    n00 = sum(r.startswith("phase00/") for r in written)
    n01 = sum(r.startswith("phase01/") for r in written)
    nbx = sum(r.startswith("biomex/") for r in written)
    print(f"[generate] {n00} Phase 00 + {n01} Phase 01 + {nbx} BIOMEX-DB profiles -> {PROFILES}")


if __name__ == "__main__":
    main()
