# CHANGELOG

All notable changes to this project will be documented in this file.

## 2026-11-26 — pfbxworld results; the thesis text quotes the per-fold protocol

### Results

- `pfbxworld` (job 5345820): 270/270 units, mean 11.3 h, max 13.9 h, ~3 063 CPU·h, peak 9.3 GB.
  `08_thesis_world_analysis.py --protocol per-fold --bases coretto2017,biomex2021`: WORLD Part B
  on BIOMEX-DB 0 of 84 significant (c058f4978).

### Thesis text (`documentation/00-thesis/monography/`)

- `thesis_build_text_numbers.py` reads both protocols: new required `--perfold-*` and
  `--pfbiomex-*` arguments; keys `pf01.*`, `pfsig.*`, `pfbx.*`, `pfbxsig.*`, `world.pf.*`,
  the stability of the folds' choices (`*.sel.*`) and per-fold minus fixed EER
  (`pf01.vsfixed.*`); claims of sentences that left the text removed, per-fold claims added
  (79 hold); `TextNumbersPerFold` tests (ea4a03cb9).
- Ch. 8-9: the per-fold protocol is the method and the primary result; new sections on what
  the folds chose and on the comparison with the fixed configuration (which keeps the
  CV-scheme and standardization ablations); BIOMEX-DB replication freezes the method, not
  the choices; Phase 01 flowchart, listing and pipeline figures redrawn (19ae511f2).
- Ch. 10, both abstracts, the ch. 7 state-of-the-art rows and appendix B on the per-fold
  numbers; `issues.md` item 5 resolved (77b73d98f).
- Findings that changed: on Coretto no verdict; against the fixed configuration the voice
  EER falls and 7 of 8 fusion cells rise; the handcrafted extractor is ahead on voice and
  overt-speech EEG, PCA on imagined-speech EEG (lowest EER, unseen phrases); on BIOMEX-DB
  late fusion vs voice is no longer significant and unseen phrases significantly worsen
  voice and late fusion (Mann-Whitney, optimistic).
- `monografia.pdf` regenerated.

## 2026-11-19/25 — per-fold runs, tables and tests; thesis job monitor

### Results (`results/thesis/`, committed with the fold models)

- `perfold` (job 5331068, 300/300 units): no declared significance family changes its verdict
  against the fixed configuration; EEG chose the fixed winner in every fold, voice chose 8
  distinct extractors over 15 folds (f619bf648).
- `pfbiomex` (jobs 5335100 + 5338456, 144/144): four families change verdict (c1e4ecf1e).
- `pfworld` (job 5345042, 150/150): Part B (EER) 0 of 140 tests significant, as with the fixed
  configuration (657f8da3c).
- `pfbxworld` (job 5345820, 270/270, 2026-11-26): Part B on BIOMEX-DB 0 of 84 significant (min
  p_holm 0.716), as with the fixed configuration (c058f4978).

### Analysis scripts (`scripts/pipeline/thesis/`)

- `thesis_build_phase01_auth_tables.py` and `thesis_build_biomex_tables.py` take a required
  `--protocol fixed|per-fold`; per-fold writes `perfold_*.csv` (adds `_selection` and
  `_vs_fixed`) and `pfbiomex_*.csv`; the fixed tables are byte-identical (f619bf648, c1e4ecf1e).
- `10_thesis_perfold_significance.sh`, `11_thesis_pfbiomex_significance.sh`: the declared
  families of `03_`/`04_` on the per-fold names; `thesis_cross_profile_significance.py` pairs a
  per-fold profile on its flat outer folds.
- `08_thesis_world_analysis.py --protocol per-fold --bases ...`: Part B only (a profile has no
  single extractor for the margin); writes `world_eer_perfold.csv`, `sig_pf*world_*.csv`.
- `thesis_grid_status.py` + `thesis_grid_watch.sh`: one ssh session, per job units done/failed
  and a job bar counting folds, per task folds, bar, measured pace per fold, time left, memory
  and limit warnings; markers from another deploy are flagged (de979e3d5, d576a52cd, e30d1e903,
  50cb794f9).
- `thesis_gridunesp.sbatch`: `--mem=16G` (pfbiomex degraded-voice units peaked at 9.3 GB and were
  killed at 8 GB, 423acdf4a); pfbxworld documented as `--array=0-269`, one unit per task (a unit
  takes ~11 h, 93a8a4739).

### Documentation

- Measured per-phase costs (Running Thesis Profiles), the per-fold branch of the results-to-text
  chain (Thesis Document), the thesis rules of the GridUnesp guide (cache warm-up, no redeploy
  mid-campaign, 16 GB, the monitor), the protocol page's status, the Home index.

## 2026-11-18/19 — per-fold protocol; feature cache

### Thesis experiment (`src/experiments/thesis/`)

- Per-fold protocol (pre-registered in `.wiki/Experiments/Thesis-Per-Fold-Protocol.md`), inside
  the binary: a profile's `selection` block makes every outer fold rank the Phase 00 candidates
  by $D_{pen}$ on its own training speakers, search a 36-point classifier grid with an inner
  4-fold split, refit the winner and score the test speakers once (`ThesisPerFold.{hpp,cpp}`;
  c98d187d8, bed1f603d). Outputs: `"protocol"` in the summary, `fold_models[f].selection`,
  `e05_<tag>_selection_{extractors,grid}.csv`.
- New profile directories `perfold/` (100), `pfbiomex/` (48), `pfworld/` (150), `pfbxworld/` (90);
  every fixed-configuration profile now declares `"selection": {"enabled": false}`.
- `FeatureDiskCache` and `thesis --warm-feature-cache` (271cbb5b1): the handcrafted candidates of
  each data view are computed once and read by every per-fold unit; key = revision + data
  fingerprint + signal + extractor block, checksummed, atomic writes; a miss in read mode stops.

### Cluster scripts

- `thesis_gridunesp.sbatch`: phases `perfold`, `pfbiomex`, `pfworld`, `pfbxworld` (read the
  cache), `THESIS_UNITS` for a trial run of named units (74b805873).
- `thesis_warm_feature_cache.sbatch`: the warm array, 11 data views.
- First runs: timed debug units (2026-11-18, deploy 74b80587), warm array 5331039 and `perfold`
  array 5331068 (2026-11-19).

### Documentation

- Running Thesis Profiles § 10 and § Feature cache, Re-run Runbook, the thesis README, the
  scripts README and How-To-Change-Anything describe how to run and change the per-fold protocol.

## 2026-11-17 — full read of the thesis; WORLD grid results; floor test

### Thesis text (`documentation/00-thesis/monography/`)

- Full read of the PDF; findings in `issues.md` § "Leitura integral do PDF", corrected in three
  commits: group A (8280b0599, the text describes the method that ran: the six descriptors per
  sub-band group, how Mel/Bark/LFCC/MFCC/BFCC are realized on the DTWPT groups, the DSNN readout
  as the mean of the logits over T), group B (3ab195841, statements no stronger than the data:
  energy *estimated* from counted spikes, non-significance not read as equivalence, the
  replication as one round without re-selection) and group C (b7065ecdf, form and bibliography).
- Items only the author can close are marked `\pendenteMarca{[PENDENTE: …]}` (e6bc63aa9): catalog
  record, examining board, defence date, dedication, acknowledgements, epigraph, the source of
  the Mel formula, the Mastrangelo (2019) chapter, the Web of Science record, the Fig. 48 image.
- `\dec` (tables read from CSV) prints a true minus sign.

### Changed — generators

- `thesis_build_text_numbers.py` takes a required `--profiles-manifest`: the profile counts the
  pipeline figure quotes (`pilot.n.total`, `world.n.co`, `world.n.bx`) are counted from
  `profiles.manifest`, and a missing or empty set is refused.
- `thesis_build_phase01_auth_tables.py` labels the extractors of `tables/phase01_extractors.csv`
  as the Phase 00 tables do ("Haar + Linear (c1)", "ANN-AE (base)", "PCA (k = 8)"); an unknown
  size or model is refused.
- `ThesisDsnnClassifier.hpp` and `Experiments/Thesis.md`: the readout is described as what the
  code computes, the mean over T of the final `Linear`'s output (analog logits), not a spike count.

### Fixed — CI

- CI's cppcheck 2.19 (Ubuntu 26.04) reports `constParameterReference` on `row_pointers` in
  `VoiceDysphonia.cpp` (WORLD writes through the returned pointers); suppressed on that line with
  the reason (5ad6874c7, whose message says 2.13 by mistake). CI green on b7065ecdf.
  `Guides/Static-Analysis.md` records the case.

### Added — WORLD grid results and the floor test

- Coretto grid (array 5328285, 150 profiles) and BIOMEX-DB grid (array 5328575, 90 profiles, on
  the rebuilt base), Part A features (`results/thesis/world_margin/`) and
  `08_thesis_world_analysis.py` → `results/thesis/world_analysis/`. Coretto: 9 of 14 levels move the
  per-utterance margin beyond resynthesis; BIOMEX-DB: none, so no level replicates; EER: 0 of 224.
- `09_thesis_world_floor.py` (rule committed first, 1fd75d377): the BIOMEX-DB null is not a
  floor effect, r = +0.14, p = 0.218, of the sign opposite to the prediction.
- BIOMEX-DB replication re-run on the base without the 3 WORLD-unanalysable pairs (array 5328574,
  6294 pairs): EERs moved by at most 0.0053; every prose claim holds.
- Thesis ch. 8–10, abstract and appendix B updated with generated numbers (`world.*` keys).

## 2026-11-16 — WORLD grid profiles; BIOMEX-DB leaves out 3 voices

### Added

- The WORLD dysphonia grid, its analysis rule committed before any run (438c38df6):
  `profiles/world/p01w_*` (Coretto, 5 voice-reading sources x 2 scopes x (control + 14 levels) =
  150) and `profiles/bxworld/bx01w_*` (BIOMEX-DB, 90). Every level is read against the control of
  its own cell. The generator takes `--calibration` (its sha256 goes in the manifest);
  `thesis_gridunesp.sbatch` gains `THESIS_PHASE=world` (0-149) and `bxworld` (0-89), one whole
  profile per task (31b0235b8: the extraction and resynthesis run once per profile).
- `scripts/data/world_voicing.py`: which voices WORLD's Harvest cannot analyse.

### Changed

- BIOMEX-DB leaves out the 3 voice windows Harvest finds unvoiced (M006 "siete", two short M014
  "seis"; `biomex_to_sqlite.py UNANALYSABLE_VOICES`); `biomex_verify.py` rebuilds each and requires
  the WORLD path to refuse it and to accept every stored voice. The base has 6294 pairs; the other
  pairs are byte-identical.

### Fixed

- `gridunesp_deploy.sh` resumes a base upload a dropped connection interrupted (`--partial-dir`),
  so a job never opens a truncated base.

## 2026-11-15 — WORLD dysphonia profiles, Praat calibration, identity tests

### Added

- `degradation.profile` selects the voice-degradation generator: `whisper_mix` (the previous one)
  or the WORLD profiles `resynthesis_control`, `rough`, `tremor`, `voice_breaks`, `diplophonia`
  (`VoiceDysphonia.cpp`; WORLD vendored at c794856b). Each profile declares only the keys it reads.
  The 64 manifest profiles with an enabled degradation now declare `whisper_mix`, which changes
  their config fingerprint (map in `Experiments/Thesis-Profile-Fingerprints.md`).
- `thesis_voice_dysphonia_gtest`: closed-loop calibration on a synthetic vowel, control
  transparency, determinism, concurrency, refusals.
- `05_thesis_dysphonia_calibration.py`: the rough/tremor requests calibrated on 134 real vowels,
  measured with Praat 6.1.38 (Parselmouth 0.4.7) → `results/thesis/dysphonia_calibration/`.
- Identity pilot (`profiles/pilot/`, `06_thesis_identity_pilot.py`, rule committed first):
  INCONCLUSIVE (15 dependent EER pairs cannot reach the ±0.01 margin).
- Powered identity test (`tools/thesis_features`, `07_thesis_identity_margin.py`, rule committed
  first): `resynthesis_control` is NOT transparent (margin shift −0.0976 against the whisper l25
  yardstick −0.0273). Every WORLD profile is therefore read as "profile minus control".
- Proposal page `Experiments/Thesis-Voice-Degradation-Proposal.md` (decisions, rules, results);
  bibliography: Praat, Parselmouth, WORLD, Kreiman and Gerratt 2005, Englert et al. 2017.

## 2026-11-14 — every data base in one directory

### Changed

- The bases moved from `~` into `~/Documentos/academico/UNESP/doutorado/databases` (the directory the Meeting01 corpora were already
  in): `database.sqlite` with its `.e05dscache` caches and `.mirror-verified` stamp,
  `biomex.sqlite` with its `.verified` stamp (sha256 re-checked after the move), `coretto_zenodo/`
  and `biomex/`.
- All 945 thesis and paraconsistentGA profiles read `dataset.root` from there (only that line
  changed in each; the profile generator and the smoke-profile maker produce the same tree, and
  `profile_audit` still passes). `paraconsistent_ga_gtest` pins the new path.
- Scripts that download, convert or verify a base default to that directory
  (`scripts/data/databases_dir.py`): `biomex_download.py`, `biomex_to_sqlite.py`,
  `biomex_verify.py`, `biomex_inventory.py`, `coretto_verify_mirror.py`, `thesis_base_facts.py`,
  `thesis_ae_input_reference.py`; an explicit path on the command line still wins.
  `ensure_datasets.sh` already used it.
- `gridunesp_deploy.sh` sends the thesis bases to `~/Documentos/academico/UNESP/doutorado/databases/`
  on the cluster (created if missing) instead of `~/`. **The next thesis deploy re-sends them**
  (about 2.3 GB + two 1.6 GB caches + 1.8 GB), and the copies at the cluster's `~/` are no
  longer read.
- `scripts/testing/test_databases_dir.py` (run by CI) keeps the directory equal in the helper,
  the generator, the deploy script and `ensure_datasets.sh`, and checks every tracked profile.
- `biomex/` was deleted after the new base was verified (see below): `biomex.sqlite` plus its
  stamp are all the thesis reads; to regenerate it, run `biomex_download.py` again.

### Added — the six excluded BIOMEX volunteers are in the base, flagged, never processed

- `biomex.sqlite` format `thesis-paired-v2`: `speaker.excluded` (0/1) and `speaker.excluded_reason`.
  F007, M001, M007, M010, M018 and M024 are stored as speakers 46..51 with 560 pairs (M001 and
  M007 get none: two complete recordings with nothing to choose between, and a 1.96 s digit
  gap), each reason saying why the strict rule dropped them and what could still be paired.
- Every reader selects `excluded = 0`, with a comment on why: `ThesisDataset.cpp` (`load_paired`),
  `biomex_verify.py`, `thesis_base_facts.py`. The strict rule is unchanged, and so are the kept
  ids (speakers 1..45, pairs 1..6297): the 6297 kept pairs are byte-identical to the previous
  base (sha256 of every row compared), the stamped facts (`results/thesis/base_facts.csv`) are
  byte-identical, and the verifier passes again (6297 equal, 0 differences; new stamp sha256
  `2a4aaf29…`). **The next thesis deploy re-sends this `biomex.sqlite` to the cluster.**
- Tests: C++ `ExcludedSpeakersAreNeverRead`; `test_biomex_convert.py` (excluded volunteers stored,
  flagged, numbered after the kept, ambiguous session gets no pairs) and
  `test_thesis_base_checks.py` (an excluded speaker's pair must not move any fact).

## 2026-11-14 — loaders tests: no hard-coded path, no skip

### Changed

- `loaders_gtest` held seven tests of real corpora that read `/home/<user>/Documentos/...` and
  called `GTEST_SKIP()` when it was absent, so in CI all seven "passed" as skipped. They moved to
  `loaders_corpus_gtest` (suites `CorpusFsdd`, `CorpusAudioMnist`, `CorpusMitBih`,
  `CorpusEegmmidb`, `CorpusSiena`): the data root is `NN_TEST_DB_ROOT`, a test whose corpus is
  missing FAILS naming the directory, and ctest registers a suite only when configured with
  `-DNN_TEST_DB_ROOT=<dir>` and that corpus exists under it (the configure log lists each). The
  three synthetic-EDF tests stay in `loaders_gtest` and always run.
- `.claude/commands` and `.github/skills`: removed the stale `guayaquil` references (the old name
  of meeting01). Besides the names, the text described things that no longer exist (the deleted
  `article-*.json` profiles and their pipeline, an `alpha_values` sweep, per-profile timings) and
  now describes the LOSO pipeline, the GA search and the per-window normalization as they are.

## 2026-11-13 — static analysis at zero

### Fixed

- `.clang-tidy` ran 131 of the intended ~400 checks: its `Checks` string was a folded block with
  `#` comments inside, which YAML keeps as text, so every glob after a comment matched nothing
  and only `clang-analyzer-*` ran. Rewritten (comments outside the string); `WarningsAsErrors: "*"`.
- Defects the full check set found:
  - the tensor comma initializer (`Tensor t; t << 1, 2, 3;`) silently accepted too few or too
    many values; it now throws `std::invalid_argument` (and stays quiet during unwinding);
  - `countShardRows` (`mat_file_utils.cpp`) ran a shell python command through `popen` and read a
    missing `count` as 0; it now parses the JSON with nlohmann and requires each count;
  - `read_csv_signal` read an empty or non-numeric field as 0; it now throws;
  - `generate_autoencoder_spike_data` drew from a global `random_device` engine; it takes a seed;
  - `NetworkSerializer::loadParametersInto` swallowed a missing LIF parameter (`try {} catch {}`)
    and kept the constructed value: a different network from the one saved. Now required;
  - meeting01: `extract_latent_size` fell back to 16 and the layer-width parsers swallowed a bad
    width; both throw, naming the key to set;
  - `reduce_time_major_output` tested `time_steps <= 0` after a cast to unsigned (never true);
  - the SQLite test helper ignored a failing step; `analysis-all` passed when a tool was absent.
  - the `autoencoder_leakyrelu` spike demo pretended to load NPZ weights (a stub that printed
    "Weights not found" and fell back to Kaiming init); it now just initializes, as documented.
- `thesis_output_gtest` wrote `e05_output_test/` into the source tree; it now uses a temp dir.

### Removed

- meeting01's LaTeX/pgfplots exports (`write_latex_exports`, the `.dat` writers, five CSV writers
  that wrote 0.0 for a missing family) and their tests: nothing read them.
  `dataset.latex_data_dir` must now be empty.
- `scripts/ci/validate_static_analysis.py`, the unread `.cppcheck` file and the never-run
  `software/nn/.github/workflows/` (GitHub runs only the repository-root workflows).

### Changed

- `analysis-cppcheck` reads `compile_commands.json` (each file with its real include paths) and
  fails on any finding. `useStlAlgorithm` and `shadowFunction` are off on purpose; false
  positives carry an inline `cppcheck-suppress` with the reason. See
  `.wiki/Guides/Static-Analysis.md`.
- CI runs clang-tidy on `software/nn/src/` only (not the vendored `_deps` sources) and prints the
  cppcheck findings when that step fails. CI's cppcheck (2.13, Ubuntu 24.04) reports findings
  2.22 does not; reproduce it with the `ubuntu:24.04` container (Static-Analysis.md).
- Tests check an optional with `if (!x.has_value()) { GTEST_FAIL() << ...; }` instead of
  `ASSERT_TRUE(x.has_value())`: CI's clang-tidy 18 does not model the macro and reported 32
  unchecked accesses after it (same semantics; Static-Analysis.md).

## 2026-11-13 — legacy experiments removed

### Removed

- The legacy experiments `autoencoderRunner` (was 03, with its grid scripts and profiles),
  `waveletAE` (was 02) and `paraconsistentBaseline` (was 00), the shared `experiments_config`
  library they used, `scripts/data/e03_dedup_master_table.py`, and their wiki pages
  (`Experiments/AutoencoderRunner.md`, `WaveletAE.md`, `ParaconsistentBaseline.md`,
  `Guides/Grid-Runbook.md`). None of them feeds the thesis or the paper; git history keeps them.

### Changed

- `meeting01_lib` is defined in `src/experiments/meeting01/CMakeLists.txt` (it was inside the
  autoencoderRunner build file) and links the core libraries directly. `thesis_lib`, the
  `nn_microscope` bindings and the autoencoder demo link `autoencoder_models` instead of
  `autoencoderRunner_lib`.
- The tests of core models that lived with autoencoderRunner moved to the core:
  `src/core/models/autoencoder/tests/autoencoder_models_gtest.cpp` (suite
  `AutoencoderModelsTest`, was `AutoencoderRunnerRedesignTest`; it guards the thesis SNN-AE
  firing-rate regularization) and `src/core/models/lstm/tests/lstm_autoencoder_training_gtest.cpp`.
- The PGO example workload is the thesis `debug.json` profile.

## 2026-11-13 — documentation overhaul

### Fixed

- `waveletAE` no longer falls back to synthetic samples when its data cannot be read (its
  hard-coded placeholder paths meant it always did), and no longer skips a wavelet that
  yields no window: both are errors. The data paths are the required spec keys
  `data.eeg_path` / `data.audio_path` (`src/experiments/waveletAE/spec.example.json`); the
  spec path is a required argument.
- `waveletAE` and `paraconsistentBaseline` printed nothing, errors included: their
  `StreamRedirector` suppresses the Logger's console output until a progress printer drains
  it, and neither has one. Removed.
- `run_thesis_profiles.sh`: new `biomex` scope; `start-over-all` runs phase00 → phase01 →
  biomex; `all` no longer includes the BIOMEX-DB profiles (sorted first, they ran before
  Phase 00); with no argument in a pipe it refuses instead of defaulting to `all`.
  `run_thesis_smoke.sh` gains `biomex`.
- `progress_gtest` is registered with `gtest_discover_tests` like every other suite (it was
  one opaque `add_test`).
- Removed two dangling symlinks (`src/protocols/`) and the unused thesis draft `backup.tex`.

### Documentation

- New entry points: repository `README.md`, `software/nn/README.md`, `documentation/README.md`,
  `notebooks/README.md`, `src/experiments/paraconsistentGA/README.md`; new wiki guides
  `How-To-Change-Anything`, `Data-Bases`, `Thesis-Document`; every `src/core/*/README.md`
  rewritten from the tree; `scripts/README.md` and the thesis experiment README rewritten.
- Stale paths (`include/nn/…`, `docs/…`, deleted scripts), a merged citation (Guido 2019 was
  given the Coretto DOI), the non-existent `nn_gtest_discover_tests` wrapper, and the
  hand-edited `Layers.hpp` instructions were corrected.
- CI runs the five Meeting01 pytest suites.

## Unreleased (2026-05-02)

### Changed

- Trainer progress callbacks now expose fractional intra-batch progress via `TrainingState::batch_progress`, and Experiment04 batch bars advance while the active batch is still being assembled and executed instead of only jumping at batch completion.
- Tensor frontend decoupling: `include/nn/tensor/Tensor.hpp` no longer depends on concrete backend headers (`XTensorBackend`/`OpenCLTensorBackend`) and no longer provides a concrete-backend default template argument.
- Active tensor alias now resolves through central backend selection (`nn::Backend`), preserving single-point backend binding in `include/nn/Backend.hpp`.
- OpenCL runtime verification API now takes `OpenCLTensorBackend` inputs directly, removing dependency on global `nn::Tensor` alias for backend-local verification probes.
- Added backend switchability coverage (`tensor_backend_switchability_gtest`) with a custom backend conformance test and source-contract assertions that guard `Tensor.hpp` against concrete-backend leakage.
- Added CMake backend-reference enforcement (`cmake/BackendImplementationGuard.cmake`): build now generates a guard translation unit that emits a compile-time error with the exact message `Backend implementation must only be refereced inside include/nn/Backend.hpp !` when concrete backend tokens are used outside allowed boundaries.

- OpenCL linear algebra: added a tiled local-memory kernel for direct
  `A^T * B` computation (`matmul_lhs_transposed_kernel`) and integrated it
  through `OpenCLTensorBackend::matmul_lhs_transposed(...)`.
- Linear backward (`dL/dW`) on OpenCL now uses the tuned lhs-transposed
  primitive instead of materializing transpose + matmul in the hot path.
- Benchmark coverage was extended to include explicit comparison between
  the new lhs-transposed path and the prior transpose-based route.

### Validation

- Added/updated targeted correctness coverage for lhs-transposed matmul in
  OpenCL backend tests.
- Long-run benchmark samples (`NN_TENSOR_BENCH_ITERS=20`) showed stable
  speedup for grad-weight computation on the active AMD rusticl stack:
  roughly 1.76x to 1.83x faster than the transpose-based probe.

## Unreleased (2026-03-17)

- Refactor: Move experiment pipeline into `LstmAutoencoderExperiment` class
  - Integrated the LSTM autoencoder runner into `autoencoderRunner`; new API in
    `src/experiments/autoencoderRunner/lib/include/meeting01.hpp` and
    `src/experiments/autoencoderRunner/lib/src/meeting01.cpp`.
  - Folded former `src/experiments/04` library, tests, and profiles into
    `src/experiments/autoencoderRunner/lib`, `src/experiments/autoencoderRunner/tests`, and
    `src/experiments/autoencoderRunner/profiles`.

- Prefetcher: `BatchPrefetcher` redesigned to single-producer thread with
  a bounded deque to serialize MAT I/O and avoid concurrent `matio` reads.

- Dataset summary: added `printDatasetSummary()` with a fast `AudioWithEEG`
  estimate (uses `min(audio_rows, eeg_rows)` by default); optional
  `--exact-summary` can compute exact counts at higher I/O cost.

- Progress UI: added in-place single-line progress helper (`progress.hpp/.cpp`)
  with effective-total logic and display capping.

- Formatting: updated `.clang-format` to allow more aggressive breaking of
  function return types and parameter lists; applied `clang-format` across
  tracked `.cpp`/`.hpp` files.

- Documentation: updated `.github/copilot-instructions.md` to reflect
  the refactors, threading and I/O constraints, and repository conventions.

### Notes
- Public header changes and API-breaking changes require a changelog entry
  and a migration guide; consider bumping the project's semantic version.
- Recommend reviewing and committing formatting-only changes as a separate
  commit to simplify review of behavioral changes.
## Unreleased (2026-03-20)

### Changed (2026-04-21)

- Experiment04 runner behavior unified: `src/experiments/meeting01/lib/src/meeting01.cpp`
  now defaults to the comparative SNN-vs-LSTM pipeline.
- Legacy standalone LSTM execution path was removed from Experiment04 runner;
  execution now always delegates to the comparative SNN-vs-LSTM pipeline.
- Legacy CLI aliases (`--meeting01`, `--lstm-autoencoder`,
  `--lstm-profile`, `--config`) were removed. The runner now accepts only
  comparative CLI flags (for example, `--comparative` and
  `--comparative-config`).

### Fixed (2026-04-16)

- SNN stability: `Leaky` and `LeakyIntegrator` now clamp effective membrane
  parameters (`R`, `C`) to positive minima during dynamics/gradient evaluation so
  `tau = R*C` and `beta = exp(-dt/tau)` remain numerically stable when optimizers
  push raw parameters toward non-positive values.
- `LeakyBPTT` backward consistency:
  - readout mode now computes resistance/capacitance gradients from the same
    non-spiking recurrence used in forward mode (no spike/reset reconstruction path).
  - threshold gradient now includes explicit direct and recurrent reset-path terms
    instead of the previous simplified accumulator.
  - capacitance is now a trainable parameter (`Tensor`) exposed by `params()` and
    updated in backward pass (aligned with `Leaky`/`LeakyIntegrator`).
- Surrogate safety: `ExponentialSurrogate` and `BoxcarSurrogate` now validate
  constructor hyperparameters and throw `std::invalid_argument` for non-positive
  `sharpness` / `window` to prevent divide-by-zero and degenerate gradients.

### Changed (2026-04-03)

- OpenCL backend initialization checks are now backend-owned via
  `OpenCLTensorBackend::initialize_runtime_or_throw(bool)`. Experiment entry
  points no longer perform sanitizer/runtime availability checks directly.
- OpenCL runtime activity verification (GPU busy probe + reconstruction-MSE
  workload) is now backend-owned via
  `OpenCLTensorBackend::verify_runtime_activity_or_throw(...)`, reducing
  OpenCL-specific code in `autoencoderRunner`.
- OpenCL runtime init/shutdown lifecycle is now backend-owned through
  `OpenCLTensorBackend::RuntimeScope` and
  `OpenCLTensorBackend::start_runtime_scope_or_throw(...)`.

### Added

- Added integrated Experiment04 LSTM assets under the AutoencoderRunner module:
  `src/experiments/autoencoderRunner/lib/include/meeting01`,
  `src/experiments/autoencoderRunner/lib/src/meeting01`, `src/experiments/autoencoderRunner/tests`, and
  `src/experiments/autoencoderRunner/profiles/lstm-*.json`.

- Dual-branch multimodal autoencoders for `autoencoderRunner`: separate EEG and audio encoder branches
  with latent-space fusion and modality-specific decoders. Implementations live under
  `src/experiments/autoencoderRunner/lib/` (Fused & Protocol autoencoders).
  - Trainable SNN membrane parameters: membrane resistance `R` and capacitance `C` are now
  learnable scalar parameters exposed via profile fields (`ae.resistance`, `ae.capacitance`) and
  persisted with model weights.
- Transparent architecture fallback: models can automatically fall back from `DualBranchFusion`
  to `ResidualDense` when modality split hints are missing.
- Shared builder utilities: `AutoencoderBuilders.hpp` centralizes encoder/decoder construction
  helpers used across ANN and SNN autoencoders.
- `reset_state()` support and proper membrane state clearing added to SNN modules (Leaky layers
  and Sequential wrappers).
- Redesign test suite: added `src/experiments/autoencoderRunner/tests/AutoencoderRedesign_gtest.cpp` with tests
  covering dual-branch and dense fallback modes for ANN and SNN.

### Changed

- `autoencoderRunner` model construction now infers protocol/eeg/audio splits and auto-selects
  `DualBranchFusion` when applicable. See `src/experiments/autoencoderRunner/lib/src/autoencoderRunner.cpp`.
- Refactored `ProtocolAutoencoder` and `ProtocolSpikingAutoencoder` to support both
  dual-branch and dense fallback execution paths.

- I/O and config formats: runtime ingestion of YAML and `.npz` (NumPy) files has been
  disabled in this branch in favor of JSON configuration. The project now vendors
  `nlohmann::json` and expects experiment configs as `.json` files (e.g. `spec.json`,
  `config.json`). The `cnpy` library remains vendored for tooling and future use,
  but runtime NPZ loading paths are guarded or disabled to avoid silent runtime
  dependencies on `.npz` artifacts.

### Fixed

- Implemented `Leaky::reset_state()` to correctly zero internal membrane state, resolving
  state-leak issues across batches and enabling deterministic reset behavior in SNN tests.

### Notes

- Validation: focused regression and redesign suites passing locally (redesign tests + Leaky
  and serializer regression slice). Static analysis (cppcheck, flawfinder) reported no new
  issues related to these changes; existing findings are in unrelated data-loader tests.
- Recommended reviewer focus: `src/experiments/autoencoderRunner/lib/*`, `include/nn/layers/Leaky.hpp`, and
  `src/experiments/autoencoderRunner/tests/AutoencoderRedesign_gtest.cpp` for behavioral review.

