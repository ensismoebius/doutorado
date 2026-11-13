#!/usr/bin/env python3
"""thesis_make_smoke_profiles.py — Generate smoke-test copies of every Experiment05 profile.

Mirrors src/experiments/thesis/profiles/{debug.json,phase00/*,phase01/*,biomex/*} into
src/experiments/thesis/profiles/smoke/ with the SAME code-path-selecting fields
(strategy, wavelet, scale, cepstral, modality, fusion_mode, classifier.type,
nested_cv, standardize_features, …) but tiny run parameters, so a run exercises
every profile's code path quickly and surfaces runtime errors that compilation
cannot catch.

Run:
    software/nn/.venv/bin/python software/nn/scripts/testing/thesis_make_smoke_profiles.py
Then drive them with scripts/testing/run_thesis_smoke.sh.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.normpath(os.path.join(HERE, "..", "..", "src", "experiments", "thesis", "profiles"))
SMOKE = os.path.join(PROFILES, "smoke")


def smoke_overrides(prof):
    """Shrink run parameters; keep every code-path-selecting field intact.

    Profiles are strict (ThesisConfig refuses a key the run does not read), so a value is
    only shrunk where the source profile declares it — e.g. a handcrafted Phase 00 profile
    has no training block, and one repeat has no seed_deterministic.
    """
    exp = prof["experiment"]
    exp["repeats"] = 1
    exp.pop("seed_deterministic", None)  # read only when repeats > 1
    exp["run_tag"] = "smoke_" + exp["run_tag"]

    # Phase 01 trains a classifier with speaker-disjoint (GroupKFold) folds and a
    # text split, so its subsets need enough distinct speakers to form 2 folds;
    # Phase 00 only ranks features and can use a tiny cap (60 > 32 + 1, the largest PCA k).
    # Truncation is round-robin across subjects (see ThesisDataset), so a small cap
    # still spans every speaker — enough for speaker-disjoint nested folds.
    runs_classifier = prof["classifier"]["enabled"]
    ds = prof["dataset"]
    ds["max_samples"] = 120 if runs_classifier else 60
    if ds["source"] == "biomex2021":
        # 51 speakers: the same ~8 samples per speaker as 120 gives the 15 Coretto speakers.
        ds["max_samples"] = 8 * 51
    ds["results_dir"] = "results/thesis/smoke"

    tr = prof.get("training")
    if tr is not None:  # absent when nothing trains (handcrafted / PCA Phase 00)
        # The cosine control has a training block without epochs or batches: only shrink
        # what the profile declares.
        for key, value in (("epochs", 2), ("samples_per_batch", 4), ("k_folds", 2),
                           ("early_stop_patience", 1)):
            if key in tr:
                tr[key] = value
    return prof


def main():
    # Wipe the mirror first so removed/renamed source profiles don't leave stale
    # smoke copies behind.
    import shutil
    shutil.rmtree(SMOKE, ignore_errors=True)

    # debug.json is a gitignored, user-local profile: only mirror it when
    # present so a fresh checkout still generates smoke profiles.
    sources = []
    debug_profile = os.path.join(PROFILES, "debug.json")
    if os.path.exists(debug_profile):
        sources.append(debug_profile)
    sources += sorted(glob.glob(os.path.join(PROFILES, "phase00", "*.json")))
    sources += sorted(glob.glob(os.path.join(PROFILES, "phase01", "*.json")))
    sources += sorted(glob.glob(os.path.join(PROFILES, "biomex", "*.json")))

    n = 0
    for src in sources:
        rel = os.path.relpath(src, PROFILES)  # e.g. phase00/p00_..json or debug.json
        dst = os.path.join(SMOKE, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src) as f:
            prof = json.load(f)
        smoke_overrides(prof)
        with open(dst, "w") as f:
            json.dump(prof, f, indent=2)
            f.write("\n")
        n += 1
    print(f"wrote {n} smoke profiles under {SMOKE}")


if __name__ == "__main__":
    main()
