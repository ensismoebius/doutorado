# scripts/

All scripts are organized into five subdirectories by purpose.

```
scripts/
  pipeline/   paper/thesis generation chains (e04 article, e05 phase00→phase01)
  data/       dataset handling & format conversion
  ci/         CI gates (called by ci.yml and coverage builds)
  dev/        developer workflow & tooling
  testing/    ground-truth parity refs + Experiment05 smoke/profile runners
  requirements.txt
  README.md   (this file)
```

---

## pipeline/ — Paper/thesis generation chains

### e04 — Guayaquil article

Full chain: `e04_run_article_profiles.sh` → CSVs → `e04_build_lstm_vs_snn_paper_data.py` → DAT files → `pdflatex`

| Script | Role |
|---|---|
| `e04_run_article_profiles.sh` | Run all 4 article profiles; calls `e04_build_lstm_vs_snn_paper_data.py` when done |
| `e04_run_backend_comparison.sh` | Run CPU vs OpenCL backend comparison |
| `e04_build_lstm_vs_snn_paper_data.py` | Aggregate `*_comparative_metrics.csv` → pgfplots DAT files |

Quick start:
```bash
cd software/nn
./scripts/pipeline/e04_run_article_profiles.sh
```

### e05 — Thesis phase00 → phase01 chain

Full chain: `run_e05_profiles.sh phase00` (scripts/testing/) → `e05_phase00_rank.py` → `winners.json` → `e05_apply_winner.py` → `run_e05_profiles.sh phase01`

| Script | Role |
|---|---|
| `e05_phase00_rank.py` | Read `results/phase00/*_summary.json`, pick the per-signal paraconsistent winner, write `winners.json` |
| `e05_apply_winner.py` | Inject `winners.json`'s winning `feature_extraction` block into the phase01 profiles' placeholder |
| `e05_build_phase00_paraconsistent_tables.py` | Generate the thesis's ranked phase00 comparison tables (`tables/phase00_*.csv`) from `results/phase00/*_summary.json`, consumed by `chapters/09-testsAndResults.tex` |

Tests for the first two live in `scripts/testing/test_e05_phase_scripts.py` (see below) — kept out of `pipeline/` since it's a test, not a pipeline step.

---

## data/ — Dataset handling & format conversion

| Script | Role |
|---|---|
| `import_mat_dataset_to_sqlite.py` | Convert `.mat` files to SQLite database |
| `sqlite_reader.py` | Read and inspect SQLite datasets |
| `npz_to_pytorch.py` | Convert `.npz` model artifacts to PyTorch `.pt` |
| `e03_dedup_master_table.py` | Remove duplicate rows from master results table |
| `verify_sqlite_roundtrip.py` | Verify MAT → SQLite roundtrip fidelity |
| `verify_sqlite_full.py` | Full integrity check on SQLite database |

---

## ci/ — CI gates

Called from `.github/workflows/ci.yml`. Do not move or rename without updating the workflow.

| Script | Role |
|---|---|
| `validate_static_analysis.py` | Check cppcheck XML against approved suppressions allowlist |
| `check_core_coverage_gate.sh` | Fail CI if core lib line coverage < 100% |
| `run_ci_docker.sh` | Run full CI pipeline inside Docker |
| `collect_external_logs.sh` | Gather logs from external build steps |

List approved cppcheck suppressions:
```bash
python3 scripts/ci/validate_static_analysis.py --list-approved
```

---

## dev/ — Developer workflow & tooling

| Script | Role |
|---|---|
| `clang-format-changed.sh` | Format staged C/C++ files (used by pre-commit hook) |
| `export_wiki_for_anytype.sh` | Export `.wiki/` as a clean Markdown folder for Anytype import |
| `sync_cross_project_skills.sh` | Publish skill catalog to other projects and global Claude dirs |
| `analyze_experiment03_grid_search.sh` | Exp03 grid search analyzer (one-off, run from repo root) |
| `apply_header_style.py` | Apply file header style to C++ sources |
| `check_core_coverage.py` | Check coverage report locally (CI uses `check_core_coverage_gate.sh`) |
| `dual_agent_consensus.sh` | Cross-check a proposer/antagonist agent report pair (SHA256 + declared checks) before accepting a task as done |

### Enable the clang-format pre-commit hook

```bash
chmod +x scripts/dev/clang-format-changed.sh
cp .githooks/pre-commit-template .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

The hook formats staged C/C++ files and re-adds them to the index before commit.

### Cross-project skill sync

```bash
bash scripts/dev/sync_cross_project_skills.sh
```

Publishes `.claude/commands/*.md` and `.github/skills/*` to global Claude and OpenCode dirs.

### Wiki export for Anytype

```bash
bash scripts/dev/export_wiki_for_anytype.sh --clean
```

Default output: `out/anytype/wiki/`. Options: `--source`, `--output`, `--include-graphify-out`, `--dry-run`.

---

## testing/ — Ground-truth parity refs & Experiment05 runners

Full docs for the PyTorch/PyWavelets parity fixtures: `testing/README.md`.

| Script | Role |
|---|---|
| `gen_pytorch_refs.py` | Regenerate `pytorch_refs.npz` fixtures consumed by `pytorch_parity_gtest` (needs `torch`) |
| `gen_pywt_refs.py` | Regenerate PyWavelets ground-truth fixtures for the C++ wavelet ops (needs `pywt`) |
| `e05_make_smoke_profiles.py` | Mirror every Experiment05 profile into `profiles/smoke/` with tiny run parameters, same code paths |
| `run_e05_smoke.sh` | Smoke-run every profile under `profiles/smoke/` — fast, surfaces runtime errors compilation can't catch |
| `run_e05_profiles.sh` | Run the REAL Experiment05 profiles (`phase00`/`phase01`/`all`) — the actual experiment, resumable, checkpointed |
| `test_e05_phase_scripts.py` | Stdlib-unittest coverage for `pipeline/e05_phase00_rank.py` + `pipeline/e05_apply_winner.py`; also run in CI |

```bash
cd software/nn
./scripts/testing/run_e05_smoke.sh          # fast sanity pass, all profiles
./scripts/testing/run_e05_profiles.sh phase00   # the real (heavy) run
python3 scripts/testing/test_e05_phase_scripts.py
```

---

## requirements.txt

Python dependencies for all scripts. Install with:
```bash
pip install -r scripts/requirements.txt
```
