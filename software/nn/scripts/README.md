# scripts/

Every script of the project except the C++ binaries. All commands run from `software/nn/`
unless a script says otherwise. Each script's own docstring or header comment is the full
reference (why it exists, every argument, every refusal); this page says which one to open.

```
scripts/
  pipeline/          ordered chains that turn runs into paper/thesis material
    thesis/            the thesis (Phase 00 → Phase 01 → BIOMEX-DB → tables → text numbers)
    meeting01/         the conference paper (nested LOSO) + the GridUnesp cluster tooling
    paraconsistentGA/  the NSGA-II autoencoder search
  analysis/          one-off studies whose CSV a pipeline or the thesis reads
  data/              obtaining, converting and verifying the bases
  testing/           test suites, ground-truth fixture generators, the thesis profile runners
  ci/                CI gates (called by .github/workflows/ci.yml)
  dev/               developer tooling
  lib/               shell helpers sourced by other scripts
  requirements.txt   Python dependencies of all of the above
```

Conventions every script here follows: **every argument is required** (no default paths), an
incomplete or unreadable input is **refused** with exit status ≠ 0 (never skipped), and
scripts in an ordered chain carry their position in the name (`01_`, `02_`, …).

Python: `python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt`. The heavy
packages (`torch`, `snntorch`, …) are needed only by the PyTorch references and analyses;
the pipeline scripts need only numpy and scipy, and several only the standard library.

---

## pipeline/thesis/ — the thesis

Order and commands: [Running Thesis Profiles › Full pipeline](../.wiki/Guides/Running-Thesis-Profiles.md#full-pipeline)
and [Thesis Document › Results to text](../.wiki/Guides/Thesis-Document.md#results-to-text).

| Script | Role |
|---|---|
| `00_thesis_generate_profiles.py` | writes every profile under `src/experiments/thesis/profiles/` and `profiles.manifest`. `--phase00-only` before the ranking; `--winners results/thesis/phase00/winners.json --fused voice` after it (the Phase 01 and BIOMEX profiles carry the winners) |
| `01_thesis_phase00_rank.py` | reads the Phase 00 results each profile declares, picks the winner per signal and the best per family by `d_penalized`, writes `winners.json`; reports ties |
| `03_thesis_phase01_significance.sh` | the declared significance families of Phase 01 → `results/thesis/phase01/significance_*.csv` |
| `04_thesis_biomex_significance.sh` | the same families on the BIOMEX-DB replication → `results/thesis/biomex/significance_*.csv` |
| `thesis_cross_profile_significance.py` | the test engine both `.sh` call: per-fold metrics of a family of profiles vs a reference; `--pairing paired\|corrected\|independent`, Holm within the family |
| `thesis_build_phase00_paraconsistent_tables.py` | `tables/phase00_{hc,ae,pca}_{voice,eegpron,eegimag}.csv` |
| `thesis_build_phase01_auth_tables.py` | `tables/phase01_{auth,controls,dysphonia,energy,extractors}.csv` |
| `thesis_build_biomex_tables.py` | `tables/biomex_{auth,dysphonia}.csv`, each BIOMEX cell beside its Coretto cell |
| `thesis_build_text_numbers.py` | `tables/thesis_numbers.tex`: every number the prose quotes (`\tn{key}`), and `CLAIMS`, the qualitative sentences checked against the data |
| `thesis_gridunesp.sbatch` | one phase as a Slurm array on GridUnesp: `THESIS_PHASE=phase00 sbatch --array=0-63 …` (phase01: `0-141`, biomex: `0-47`) |

The profile runners themselves are in `testing/` (`run_thesis_profiles.sh`, `run_thesis_smoke.sh`).

## pipeline/meeting01/ — the conference paper and the cluster

The paper's experiment is the nested leave-one-group-out grid of `profiles/meeting01-loso.json`
(see [Meeting01](../.wiki/Experiments/Meeting01.md)). It runs for weeks, so it runs on GridUnesp
([GridUnesp Deployment](../.wiki/Guides/GridUnesp-Deployment.md)).

**The chain** (numbered by execution order):

| Script | Role |
|---|---|
| `00_meeting01_prevalidate_baselines.sh`, `00_meeting01_prevalidate_gridunesp.sbatch`, `00_meeting01_prevalidation_report.py` | the reduced-GA pre-validation of every family on the development folds (the evidence for removing LSTM/GRU/Transformer-AE from the grid), locally or as one cluster job, and its report |
| `01_meeting01_run_loso.sh` | the full grid, one process per (dataset, outer fold); needs `EXPERIMENT_CONFIRMED=1`; chains into 03 → 02 → 04 when done |
| `01_meeting01_run_loso_gridunesp.sbatch`, `submit_loso_array.sh` | the same on the cluster: one job under job-nanny, or a Slurm array of one-CPU tasks |
| `02_meeting01_build_loso_paper_data.py` | aggregate `*_comparative_metrics.csv` into the paper's pgfplots DAT files |
| `03_meeting01_pca_mean_baselines.py` | PCA and mean-frame reference baselines |
| `04_meeting01_significance_tests.py` | recording-level hierarchical significance tests |
| `05_meeting01_t1_transformer.{sh,py}` | T1: does warm-up or pre-norm get the Transformer-AE off the mean-predictor plateau |
| `06_meeting01_stopping_calibration.py` | measures the stopping rule's skill threshold |
| `07_meeting01_design_study.{sh,py}` | the design study on a logged PyTorch twin of the SNN-AE |
| `08_meeting01_budget_test.{sh,py}` | does the training budget, not the model, decide two pre-validation results |
| `09_meeting01_energy_breakeven.py` | the paper's energy break-even figure and numbers |
| `10_meeting01_learning_curves.py` | the paper's learning-curve figure (fit gap vs generalization gap) |

**Shared modules and cluster tooling** (not numbered: used by the chain, or run by hand):

| Script | Role |
|---|---|
| `meeting01_results.py` | which files of `results/meeting01/` belong to one run, and whether they can be used (shared by 02/03/04) |
| `source_revision.py`, `_provenance.sh` | which code produced a run (the cluster tree has no git metadata) |
| `ensure_datasets.sh`, `check_dataset_progress.sh` | fetch the four datasets (FSDD, AudioMNIST, EEGMMIDB, Siena) into the layout the profile expects; check a running fetch |
| `gridunesp_deploy.sh` | push this checkout to GridUnesp, bootstrap the toolchain, fetch datasets, build (`meeting01`, `thesis`), send the thesis bases |
| `gridunesp_setup_env.sh` | one-time conda toolchain on the cluster |
| `run_gridunesp_docker_sim.sh` | rehearse the cluster toolchain in a local AlmaLinux container (`Dockerfile.gridunesp-sim`) |
| `_gridunesp_env.sh`, `gridunesp_config.py` | SSH credentials and connection settings (read from `.env`, never printed) — sourced/imported, not run |
| `_job_nanny_reconcile.sh` | after a job, make `~/software/nn/results` hold what the run left, not also what it deleted |
| `monitor.py`, `remote_monitor.sh`, `pull_progress.sh` | live progress of a grid run: locally, on the login node, or on a synced copy |
| `gridunesp_tui.py`, `gridunesp_status_remote.py` | a terminal dashboard of every cluster script (queue, builds, progress, actions) and its remote half |
| `ga_cache.py`, `collect_ga_local.py` | GA per-individual cache: incremental reader, and the SSH collector |
| `dashboard/` | read-only FastAPI web dashboard (`server.py`, `static/`), with `downsample.py` and `sweep_state.py` |

## pipeline/paraconsistentGA/

| Script | Role |
|---|---|
| `01_paraconsistentGA_run_all_profiles.sh` | build, run all 12 NSGA-II population profiles, summarise the Pareto fronts ([paraconsistentGA](../.wiki/Experiments/ParaconsistentGA.md)) |

## analysis/

| Script | Writes / answers | Read by |
|---|---|---|
| `thesis_base_facts.py --biomex --coretto --out` | `results/thesis/base_facts.csv`: the base facts ch. 8 states, from the bases and their verification stamps | `thesis_build_text_numbers.py` (`base.*`) |
| `thesis_ae_input_reference.py --db --out` | `results/thesis/ae_input_reference.csv`: does the AE/PCA input carry the speaker (needs torch) | `thesis_build_text_numbers.py` (`aeinput.*`) |
| `thesis_eeg_band_power.py --db --out` | `results/thesis/eeg_band_power.csv`: where the EEG power lies relative to the 32 Hz sub-bands | quoted in `issues.md` / ch. 8 discussion |
| `compare_ann_ae_torch.py`, `compare_lstm_ae_torch.py`, `snn_gap_ablation_torch.py` | Meeting01: PyTorch twins and ablations that explain the SNN-AE gap | the Meeting01 wiki page |

## data/ — the bases

Full procedure: [Data Bases](../.wiki/Guides/Data-Bases.md).

| Script | Role |
|---|---|
| `import_mat_dataset_to_sqlite.py <root> <db>` | Coretto `.mat` (S01..S15) → `~/database.sqlite` |
| `coretto_verify_mirror.py --db --mirror` | the Coretto SQLite against the Zenodo EEG copy → stamp `<db>.mirror-verified` |
| `biomex_download.py --dest` | BIOMEX-DB from Mendeley Data (SHA-256 checked; EEG + voice only) |
| `biomex_inventory.py --src [--csv]` | what each BIOMEX volunteer folder holds (read before changing a rule) |
| `biomex_to_sqlite.py --src --out` | the paired BIOMEX SQLite (`~/biomex.sqlite`) |
| `biomex_verify.py --src --db` | the converted SQLite re-derived by a second path → stamp `<db>.verified` |
| `sqlite_reader.py`, `verify_sqlite_roundtrip.py`, `verify_sqlite_full.py` | inspect a SQLite base; older MAT → SQLite fidelity checks |
| `npz_to_pytorch.py` | convert `.npz` model files to PyTorch `.pt` |
| `e03_dedup_master_table.py` | remove duplicate rows from the autoencoderRunner master table |

## testing/

Details on the ground-truth fixtures: [`testing/README.md`](testing/README.md) and
[Ground-Truth and Smoke Testing](../.wiki/Guides/Ground-Truth-and-Smoke-Testing.md).

**Profile runners (the thesis):**

| Script | Role |
|---|---|
| `run_thesis_profiles.sh [phase00\|phase01\|biomex\|all\|start-over-all]` | the real profiles: resumable, parallel, live dashboard; no default scope |
| `run_thesis_smoke.sh [phase00\|phase01\|biomex\|all]` | the smoke mirror (tiny parameters, same code paths) |
| `thesis_make_smoke_profiles.py` | regenerate `profiles/smoke/` from the real profiles |

**Test suites** (Python; the C++ tests run through `ctest`):

| File | Tests | Runner | In CI |
|---|---|---|---|
| `test_thesis_phase_scripts.py` | the thesis pipeline scripts, and `GeneratedTree` (the tracked profiles are the generator's output) | `python3 <file>` (stdlib unittest) | yes |
| `test_biomex_convert.py` | `biomex_to_sqlite.py` on a synthetic volunteer | `python3 <file>` | yes |
| `test_thesis_base_checks.py` | `coretto_verify_mirror.py`, `thesis_base_facts.py` | `python3 <file>` | yes |
| `test_downsample.py`, `test_ga_cache.py`, `test_sweep_state.py`, `test_meeting01_dashboard_data.py`, `test_remote_ga_collect.py` | the Meeting01 monitoring tools | `python3 -m pytest scripts/testing/test_<name>.py` | yes |

**Ground-truth fixture generators** (developer steps; the fixtures are committed and the C++
parity tests read them without the reference library):

| Script | Fixture → C++ test |
|---|---|
| `gen_pytorch_refs.py` | layers vs PyTorch → `pytorch_parity_gtest` |
| `gen_micro_network_refs.py` | whole micro-networks vs PyTorch → `micro_network_parity_gtest` |
| `gen_lstm_ae_refs.py`, `lstm_ae_torch.py` | the whole LSTM autoencoder vs its PyTorch twin → `lstm_ae_pytorch_parity_gtest` |
| `gen_optimizer_refs.py` | optimizers vs their reference implementations → `optimizer_parity_gtest` |
| `gen_pywt_refs.py` | wavelet sub-band energies vs PyWavelets → `pywt_parity_gtest` |
| `gen_thesis_spectrum_refs.py` | the thesis AE/PCA input vs scipy + PyTorch → `thesis_spectrum_parity_gtest` |
| `gen_independent_refs.py` | independent cross-checks of `SpikeTimeLoss` / `PoissonLatentLayer` (not an external library: read its warning) |

## ci/

Called from `.github/workflows/ci.yml`; renaming one needs the workflow changed too.

| Script | Role |
|---|---|
| `validate_static_analysis.py` | cppcheck XML against the approved suppressions (`--list-approved` lists them) |
| `check_core_coverage_gate.sh` | fail if core line coverage is below the gate |
| `run_ci_docker.sh` | the CI pipeline inside Docker (`Dockerfile.ci`) |
| `collect_external_logs.sh` | gather logs from external build steps |

## dev/

| Script | Role |
|---|---|
| `clang-format-changed.sh` | format staged C/C++ files; install as a hook with `cp .githooks/pre-commit-template ../../.git/hooks/pre-commit && chmod +x ../../.git/hooks/pre-commit` (the `.git` is at the repository root) |
| `apply_header_style.py` | apply the file-header style to C++ sources |
| `check_core_coverage.py` | the coverage check, locally |
| `dual_agent_consensus.sh` | cross-check a proposer/antagonist agent report pair ([Dual-Agent Consensus](../.wiki/Development/Dual-Agent-Consensus.md)) |
| `export_wiki_for_anytype.sh [--clean]` | export `.wiki/` as plain Markdown (default `out/anytype/wiki/`) |
| `sync_cross_project_skills.sh` | publish `.claude/commands/` and `.github/skills/` to the global agent directories |

## lib/

| Script | Role |
|---|---|
| `run_eta.sh` | work-weighted, EMA-smoothed ETA, sourced by `run_thesis_profiles.sh` and `01_meeting01_run_loso.sh` |
