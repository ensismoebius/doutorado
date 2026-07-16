# scripts/

All scripts are organized into four subdirectories by purpose.

```
scripts/
  pipeline/   paper generation chain
  data/       dataset handling & format conversion
  ci/         CI gates (called by ci.yml and coverage builds)
  dev/        developer workflow & tooling
  requirements.txt
  README.md   (this file)
```

---

## pipeline/ — Paper generation chain

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

## requirements.txt

Python dependencies for all scripts. Install with:
```bash
pip install -r scripts/requirements.txt
```
