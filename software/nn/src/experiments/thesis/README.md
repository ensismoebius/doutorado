# thesis — the thesis experiment (was "Experiment05")

Biometric authentication of severely dysphonic speakers, helped by EEG of imagined and
pronounced speech. One C++ binary, `thesis`, runs one **profile** (a JSON file that declares
every setting; nothing has a default). The thesis is three sets of profiles run in order:

```
Phase 00  rank feature extractors, no classifier        profiles/phase00/  275 profiles
   │      (handcrafted DTWPT, ANN-AE, SNN-AE, PCA; per signal: voice, EEG pronounced, EEG imagined)
   ▼      winner per signal = lowest d_penalized  ──►  results/thesis/phase00/winners.json
Phase 01  authenticate with the winners             profiles/phase01/  142 profiles
   │      (DSNN vs its ANN twin vs cosine; voice, EEG, fusions; simulated dysphonia; ...)
   ▼
BIOMEX    replicate the Phase 01 cells on a second base  profiles/biomex/   48 profiles
```

**The protocol, its rationale and its history are in the wiki:
[`.wiki/Experiments/Thesis.md`](../../../.wiki/Experiments/Thesis.md).** How to run the
profiles: [Running Thesis Profiles](../../../.wiki/Guides/Running-Thesis-Profiles.md). How results
become thesis text: [Thesis Document](../../../.wiki/Guides/Thesis-Document.md). This README is
the map of this folder.

## Build and run (from `software/nn`)

```bash
cmake --preset=max-performance                      # once
cmake --build out/build/max-performance --target thesis -j$(nproc)
./out/build/max-performance/src/experiments/thesis/thesis \
    --config src/experiments/thesis/profiles/phase00/p00_pca_k8_voice.json   # every repetition
./out/build/max-performance/src/experiments/thesis/thesis \
    --config <profile.json> --repeat 1               # only repetition 1 (what the cluster runs)
```

The binary accepts exactly `--config` and `--repeat`; anything else is refused. It reads the
base named by the profile's `dataset.root` (`~/database.sqlite` for Coretto,
`~/biomex.sqlite` for BIOMEX-DB — see [Data Bases](../../../.wiki/Guides/Data-Bases.md)).

Never write a profile by hand: they are generated (§ "Changing it" below).

## Folder map

| Path | Role |
|---|---|
| `thesis.cpp` | entry point: reads the CLI and the profile, runs Phase 00 or Phase 01, writes the outputs |
| `lib/include/ThesisConfig.hpp`, `lib/src/ThesisConfig.cpp` | the profile schema; every field is `Required<T>` (`Required.hpp`, `StrictJson.hpp`); the table at the top of the `.cpp` says which key is read under which condition |
| `lib/include/ThesisCli.hpp` | the command line (`--config`, `--repeat`) |
| `lib/*/ThesisDataset.*`, `ThesisSample.hpp`, `TextSplit.hpp` | loading the base into per-speaker samples (with the decoded-dataset binary cache), and the seen/unseen phrase split |
| `lib/*/EegPairing.*` | which EEG trial goes with each voice trial (the same trial, or an imagined one) |
| `lib/*/ThesisFeatureExtraction.*`, `ThesisHandcraftedFeatures.cpp`, `ThesisFeatureExtractionInternal.hpp` | the extractors: handcrafted DTWPT descriptors, ANN-AE, SNN-AE (spectral input) |
| `lib/*/ThesisPca.*` | PCA, the training-free linear extractor |
| `lib/*/ThesisParaconsistent.*`, `ParaconsistentSubsets.*` | the Phase 00 score (`α`, `β`, `d_truth`, `d_penalized`) and its recomputation on speaker subsets |
| `lib/*/ThesisClassifiers.*`, `ThesisDsnnClassifier.hpp`, `ThesisAnnClassifier.hpp`, `ThesisClassifiersInternal.hpp` | Phase 01: the DSNN, its non-spiking twin `ann`, the `cosine` control, EER scoring, nested/flat cross-validation |
| `lib/*/VoiceDegradation.*` | the simulated dysphonia (LPC whisper mix; described step by step in thesis ch. 8) |
| `lib/*/ThesisOutput.*`, `ClassificationResult.hpp`, `FoldResult.hpp` | the result files |
| `lib/include/IndexIntersection.hpp` | order-independent intersection of index lists (the July fold bug) |
| `profiles/` | generated profiles: `phase00/`, `phase01/`, `biomex/`, `smoke/`, `profiles.manifest`; `debug.json` is a hand-kept quick-run profile |
| `tests/` | 12 GoogleTest files built into 6 executables (below) |

## Outputs

Each run writes into the profile's `dataset.results_dir` (`results/thesis/<phase>/`), with
`<tag>` = `e05_<profile name>_rep<k>`:

| File | Content |
|---|---|
| `e05_<tag>_summary.json` | the configuration the run used + mean/sd/CI95 of every metric |
| `e05_<tag>_metrics.csv` | one row per fold: accuracy, F1, precision, recall, EER, AUC, model path (Phase 01) |
| `e05_<tag>_paraconsistent.csv`, `..._subsets.csv` | the Phase 00 score per feature set, and on speaker subsets |
| `e05_<tag>_comparison.dat`, `..._learning_curves.dat` | pgfplots data |
| `models/<tag>/<feature label>/fold_<N>.bin` | the trained Phase 01 model of each outer fold (committed) |
| `logs/`, `.done/`, `.failed/` | written by `scripts/testing/run_thesis_profiles.sh` (per-profile log, completion markers) |

## Changing it

| I want to… | Do this |
|---|---|
| add or change a profile key | add the `Required<T>` field to `ThesisConfig.hpp`, read and validate it in `ThesisConfig.cpp` (also its row in the "who reads what" table), write it in `scripts/pipeline/thesis/00_thesis_generate_profiles.py`, regenerate the tree, run `thesis_feature_extraction_gtest` (it holds the strict-config tests) and `thesis_profile_audit_gtest` |
| add, remove or change profiles | edit `00_thesis_generate_profiles.py` and run it (`--phase00-only`, or `--winners results/thesis/phase00/winners.json --fused voice`); never edit a JSON under `profiles/`: `GeneratedTree` in `scripts/testing/test_thesis_phase_scripts.py` (CI) refuses a tree the generator did not write |
| add an extractor | `ThesisFeatureExtraction.cpp` (+ a family in `FAMILIES` of the generator and in `01_thesis_phase00_rank.py`); test it in `thesis_feature_extraction_gtest` |
| add a classifier or control | `ThesisClassifiers.cpp` (+ the classifier type in `ThesisConfig`); test in `thesis_classifiers_gtest` |
| change the dysphonia simulation | `VoiceDegradation.cpp`; then regenerate `documentation/00-thesis/monography/images/simulatedDysphonia.pdf` (its script compiles this very file) and re-check ch. 8 §"disfonia simulada" |
| change an output file | `ThesisOutput.cpp`; then the readers: `scripts/pipeline/thesis/thesis_build_*` and their tests |
| run it on the cluster | [`.wiki/Experiments/Thesis.md` → Running on GridUnesp](../../../.wiki/Experiments/Thesis.md#running-on-gridunesp-2026-11-12) |

After any change that alters results, the profiles of record must run again; see
[Re-run Runbook](../../../.wiki/Guides/Re-run-Runbook.md).

## Tests

```bash
cmake --build out/build/max-performance -j$(nproc) --target thesis_profile_audit_gtest \
    thesis_feature_extraction_gtest thesis_classifiers_gtest thesis_output_gtest \
    thesis_spike_loss_e2e_gtest thesis_freearch_gtest
for t in profile_audit feature_extraction classifiers output spike_loss_e2e freearch; do
    ./out/build/max-performance/src/experiments/thesis/tests/thesis_${t}_gtest || break
done
```

Running the six executables directly is the simplest complete run: `ctest -R` filters by
GoogleTest suite name, and the suites here do not share a prefix (`Thesis*`, `AnnTwin`,
`CosineControl`, `EegPairing`, `VoiceDegradation`, …), so `ctest -R Thesis` silently skips some.
To run one suite: `ctest --test-dir out/build/max-performance -R 'VoiceDegradation' --output-on-failure`.

`thesis_profile_audit_gtest` checks every generated profile against
`profiles.manifest` and the schema. The Python side of the pipeline is tested by
`scripts/testing/test_thesis_phase_scripts.py` (see [`scripts/README.md`](../../../scripts/README.md)).
