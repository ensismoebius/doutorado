# thesis — the thesis experiment (was "Experiment05")

Biometric authentication of severely dysphonic speakers, helped by EEG of imagined and
pronounced speech. One C++ binary, `thesis`, runs one **profile** (a JSON file that declares
every setting; nothing has a default). The thesis is these sets of profiles, run in order:

```
Phase 00  rank feature extractors, no classifier        profiles/phase00/  275 profiles
   │      (handcrafted DTWPT, ANN-AE, SNN-AE, PCA; per signal: voice, EEG pronounced, EEG imagined)
   ▼      winner per signal = lowest d_penalized  ──►  results/thesis/phase00/winners.json
Phase 01  authenticate with the winners             profiles/phase01/  142 profiles
   │      (DSNN vs its ANN twin vs cosine; voice, EEG, fusions; simulated dysphonia; ...)
   ▼
BIOMEX    replicate the Phase 01 cells on a second base  profiles/biomex/   48 profiles
   │
   ▼
WORLD     simulated dysphonia, each level minus its resynthesis control
          identity pilot (resynthesis alone)                profiles/pilot/     6 profiles
          grid on Coretto / on BIOMEX-DB                    profiles/world/   150 profiles
                                                            profiles/bxworld/  90 profiles

Per-fold  the same cells, but the extractor (by D_pen) and the classifier's hyperparameters
protocol  (inner 4-fold search, 36 points) are chosen INSIDE each outer fold, on its training
          speakers only; the primary result from 2026-11-18
                                                            profiles/perfold/   100 profiles
                                                            profiles/pfbiomex/   48 profiles
                                                            profiles/pfworld/   150 profiles
                                                            profiles/pfbxworld/  90 profiles
```

The per-fold protocol is described in
[`.wiki/Experiments/Thesis-Per-Fold-Protocol.md`](../../../.wiki/Experiments/Thesis-Per-Fold-Protocol.md).
It runs on the cluster after a warm step fills the feature cache
([Running Thesis Profiles § 10](../../../.wiki/Guides/Running-Thesis-Profiles.md#10-per-fold-protocol--extractor-and-hyperparameters-chosen-inside-each-fold)).

A second binary, `thesis_features` (`tools/thesis_features.cpp`), writes the clean and degraded
feature vector of every sample of a profile (one CSV row each), through the same loader,
extractor and degradation as `thesis`; the identity and WORLD analyses
(`scripts/pipeline/thesis/07_`, `08_`, `09_`) read them.

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
THESIS_FEATURE_CACHE=<dir> THESIS_FEATURE_CACHE_REVISION=<label> THESIS_FEATURE_CACHE_MODE=write \
./out/build/max-performance/src/experiments/thesis/thesis \
    --config <per-fold profile.json> --warm-feature-cache   # fill the feature cache, run nothing
cmake --build out/build/max-performance --target thesis_features -j$(nproc)
./out/build/max-performance/src/experiments/thesis/thesis_features \
    --config <profile.json> --out-dir <dir>          # feature CSVs, no classifier
```

The binary accepts exactly `--config`, `--repeat` and `--warm-feature-cache` (per-fold profiles only, never with `--repeat`); anything else is refused. It reads the
base named by the profile's `dataset.root` (`~/Documentos/academico/UNESP/doutorado/databases/database.sqlite` for Coretto,
`~/Documentos/academico/UNESP/doutorado/databases/biomex.sqlite` for BIOMEX-DB — see [Data Bases](../../../.wiki/Guides/Data-Bases.md)).

Never write a profile by hand: they are generated (§ "Changing it" below).

## Folder map

| Path | Role |
|---|---|
| `thesis.cpp` | entry point: reads the CLI and the profile, runs Phase 00 or Phase 01, writes the outputs |
| `lib/include/ThesisConfig.hpp`, `lib/src/ThesisConfig.cpp` | the profile schema; every field is `Required<T>` (`Required.hpp`, `StrictJson.hpp`); the table at the top of the `.cpp` says which key is read under which condition |
| `lib/include/ThesisCli.hpp` | the command line (`--config`, `--repeat`, `--warm-feature-cache`) |
| `lib/*/ThesisPerFold.*` | the per-fold protocol: outer and inner fold rows, extractor ranking per fold, the hyperparameter grid and its winner, the refit; `run_per_fold`, `warm_feature_cache` |
| `lib/*/FeatureDiskCache.*` | the on-disk cache of handcrafted candidate features shared by the per-fold processes (key = revision + data fingerprint + signal + extractor block; checksummed, atomic writes) |
| `lib/*/ThesisDataset.*`, `ThesisSample.hpp`, `TextSplit.hpp` | loading the base into per-speaker samples (with the decoded-dataset binary cache), and the seen/unseen phrase split |
| `lib/*/EegPairing.*` | which EEG trial goes with each voice trial (the same trial, or an imagined one) |
| `lib/*/ThesisFeatureExtraction.*`, `ThesisHandcraftedFeatures.cpp`, `ThesisFeatureExtractionInternal.hpp` | the extractors: handcrafted DTWPT descriptors, ANN-AE, SNN-AE (spectral input) |
| `lib/*/ThesisPca.*` | PCA, the training-free linear extractor |
| `lib/*/ThesisParaconsistent.*`, `ParaconsistentSubsets.*` | the Phase 00 score (`α`, `β`, `d_truth`, `d_penalized`) and its recomputation on speaker subsets |
| `lib/*/ThesisClassifiers.*`, `ThesisDsnnClassifier.hpp`, `ThesisAnnClassifier.hpp`, `ThesisClassifiersInternal.hpp` | Phase 01: the DSNN, its non-spiking twin `ann`, the `cosine` control, EER scoring, nested/flat cross-validation |
| `lib/*/VoiceDegradation.*` | the simulated dysphonia (LPC whisper mix; described step by step in thesis ch. 8) |
| `lib/*/VoiceDysphonia.*` | the other simulated dysphonias (WORLD: `resynthesis_control`, `rough`, `tremor`, `voice_breaks`, `diplophonia`; selected by `degradation.profile`) |
| `lib/*/DegradeVoices.*` | applies the profile's degradation to the voices; shared by `thesis` and `thesis_features`, so both degrade identically |
| `tools/thesis_features.cpp` | the `thesis_features` binary (feature CSVs of a profile) |
| `lib/*/ThesisOutput.*`, `ClassificationResult.hpp`, `FoldResult.hpp` | the result files |
| `lib/include/IndexIntersection.hpp` | order-independent intersection of index lists (the July fold bug) |
| `profiles/` | generated profiles: `phase00/`, `phase01/`, `biomex/`, `pilot/`, `world/`, `bxworld/`, `perfold/`, `pfbiomex/`, `pfworld/`, `pfbxworld/`, `smoke/`, `profiles.manifest`; `debug.json` is a hand-kept quick-run profile |
| `tests/` | 14 GoogleTest files (+ `ThesisTestConfig.hpp`) built into 7 executables (below) |

## Outputs

Each run writes into the profile's `dataset.results_dir` (`results/thesis/<phase>/`), with
`<tag>` = `e05_<profile name>_rep<k>`:

| File | Content |
|---|---|
| `e05_<tag>_summary.json` | the configuration the run used + mean/sd/CI95 of every metric |
| `e05_<tag>_metrics.csv` | one row per fold: accuracy, F1, precision, recall, EER, AUC, model path (Phase 01) |
| `e05_<tag>_paraconsistent.csv`, `..._subsets.csv` | the Phase 00 score per feature set, and on speaker subsets |
| `e05_<tag>_comparison.dat`, `..._learning_curves.dat` | pgfplots data |
| `e05_<tag>_selection_extractors.csv`, `..._selection_grid.csv` | per-fold protocol: every fold's extractor ranking and its grid search (the summary also carries them, under `fold_models[f].selection`, and says `"protocol": "per-fold"`) |
| `models/<tag>/<feature label>/fold_<N>.bin` | the trained Phase 01 model of each outer fold (committed) |
| `logs/`, `.done/`, `.failed/` | written by `scripts/testing/run_thesis_profiles.sh` (per-profile log, completion markers) |

## Changing it

| I want to… | Do this |
|---|---|
| add or change a profile key | add the `Required<T>` field to `ThesisConfig.hpp`, read and validate it in `ThesisConfig.cpp` (also its row in the "who reads what" table), write it in `scripts/pipeline/thesis/00_thesis_generate_profiles.py`, regenerate the tree, run `thesis_feature_extraction_gtest` (it holds the strict-config tests) and `thesis_profile_audit_gtest` |
| add, remove or change profiles | edit `00_thesis_generate_profiles.py` and run it (`--phase00-only`, or `--winners results/thesis/phase00/winners.json --fused voice`); never edit a JSON under `profiles/`: `GeneratedTree` in `scripts/testing/test_thesis_phase_scripts.py` (CI) refuses a tree the generator did not write |
| add an extractor | `ThesisFeatureExtraction.cpp` (+ a family in `FAMILIES` of the generator and in `01_thesis_phase00_rank.py`); test it in `thesis_feature_extraction_gtest` |
| add a classifier or control | `ThesisClassifiers.cpp` (+ the classifier type in `ThesisConfig`); test in `thesis_classifiers_gtest` |
| change the dysphonia simulation | `VoiceDegradation.cpp` (`whisper_mix`) or `VoiceDysphonia.cpp` (the WORLD profiles, calibrated by `thesis_voice_dysphonia_gtest`); then regenerate `documentation/00-thesis/monography/images/simulatedDysphonia.pdf` (its script compiles this very file) and re-check ch. 8 §"disfonia simulada" |
| change an output file | `ThesisOutput.cpp`; then the readers: `scripts/pipeline/thesis/thesis_build_*` and their tests |
| change the per-fold grid or candidates | `SELECTION_GRID` / `extractor_candidates` in `00_thesis_generate_profiles.py`; regenerate; `thesis_classifiers_gtest` (suites `PerFold*`, `FeatureDiskCache`), `thesis_profile_audit_gtest`; the pre-registration page must record the amendment |
| run it on the cluster | [`.wiki/Experiments/Thesis.md` → Running on GridUnesp](../../../.wiki/Experiments/Thesis.md#running-on-gridunesp-2026-11-12) |

After any change that alters results, the profiles of record must run again; see
[Re-run Runbook](../../../.wiki/Guides/Re-run-Runbook.md).

## Tests

```bash
cmake --build out/build/max-performance -j$(nproc) --target thesis_profile_audit_gtest \
    thesis_feature_extraction_gtest thesis_classifiers_gtest thesis_output_gtest \
    thesis_spike_loss_e2e_gtest thesis_freearch_gtest thesis_voice_dysphonia_gtest
for t in profile_audit feature_extraction classifiers output spike_loss_e2e freearch voice_dysphonia; do
    ./out/build/max-performance/src/experiments/thesis/tests/thesis_${t}_gtest || break
done
```

Running the seven executables directly is the simplest complete run: `ctest -R` filters by
GoogleTest suite name, and the suites here do not share a prefix (`Thesis*`, `AnnTwin`,
`CosineControl`, `EegPairing`, `VoiceDegradation`, …), so `ctest -R Thesis` silently skips some.
To run one suite: `ctest --test-dir out/build/max-performance -R 'VoiceDegradation' --output-on-failure`.

`thesis_profile_audit_gtest` checks every generated profile against
`profiles.manifest` and the schema. The Python side of the pipeline is tested by
`scripts/testing/test_thesis_phase_scripts.py` (see [`scripts/README.md`](../../../scripts/README.md)).
