#!/usr/bin/env python3
"""e05_make_smoke_profiles.py — Generate smoke-test copies of every Experiment05 profile.

Mirrors src/experiments/05/profiles/{debug.json,phase00/*,phase01/*} into
src/experiments/05/profiles/smoke/ with the SAME code-path-selecting fields
(strategy, wavelet, scale, cepstral, modality, fusion_mode, classifier.type,
nested_cv, standardize_features, …) but tiny run parameters, so a run exercises
every profile's code path quickly and surfaces runtime errors that compilation
cannot catch.

Run:
    software/nn/.venv/bin/python software/nn/scripts/testing/e05_make_smoke_profiles.py
Then drive them with scripts/testing/run_e05_smoke.sh.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.normpath(os.path.join(HERE, "..", "..", "src", "experiments", "05", "profiles"))
SMOKE = os.path.join(PROFILES, "smoke")


def smoke_overrides(prof):
    """Shrink run parameters; keep every code-path-selecting field intact."""
    exp = prof.setdefault("experiment", {})
    exp["repeats"] = 1
    exp["seed_deterministic"] = True
    exp["run_tag"] = "smoke_" + exp.get("run_tag", "e05")

    # Phase 01 trains a classifier with speaker-disjoint (GroupKFold) folds and a
    # text split, so its subsets need enough distinct speakers to form 2 folds;
    # Phase 00 only ranks features and can use a tiny cap.
    # Truncation is round-robin across subjects (see E05Dataset), so a small cap
    # still spans every speaker — enough for speaker-disjoint nested folds.
    runs_classifier = prof.get("classifier", {}).get("enabled", True)
    ds = prof.setdefault("dataset", {})
    ds["max_samples"] = 120 if runs_classifier else 60
    ds["results_dir"] = "results/smoke"

    tr = prof.setdefault("training", {})
    tr["epochs"] = 2
    tr["k_folds"] = 2
    tr["samples_per_batch"] = 4
    tr["early_stop_patience"] = 1
    return prof


def main():
    # Wipe the mirror first so removed/renamed source profiles don't leave stale
    # smoke copies behind.
    import shutil
    shutil.rmtree(SMOKE, ignore_errors=True)

    sources = [os.path.join(PROFILES, "debug.json")]
    sources += sorted(glob.glob(os.path.join(PROFILES, "phase00", "*.json")))
    sources += sorted(glob.glob(os.path.join(PROFILES, "phase01", "*.json")))

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
