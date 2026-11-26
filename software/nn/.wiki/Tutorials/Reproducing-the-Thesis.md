# Tutorial 3 — Reproducing the thesis results

This page takes you from a fresh clone to the numbers of the thesis
(*Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela Imagined
Speech*, `documentation/00-thesis/monography/monografia.pdf`), step by step, and names every
script and file on the way. You need no knowledge of the code. Commands run from
`software/nn/` unless a `cd` is shown.

## 1. What "reproducing" means here

Every number in the thesis was produced by a chain of files, and every link of that chain is in
this repository:

```
 <databases>/database.sqlite, biomex.sqlite     the two bases (outside git; § 3)
        │
        ▼  thesis binary, one run per profile            (months of CPU; § 6)
 results/thesis/<phase>/                         raw outputs + trained models (in git)
        │
        ▼  table, test and analysis scripts              (minutes; § 5)
 documentation/00-thesis/monography/tables/*.csv          the tables of ch. 9
 documentation/00-thesis/monography/tables/thesis_numbers.tex   every number of the prose
        │
        ▼  latexmk
 monografia.pdf
```

So there are two ways to reproduce the thesis, and they answer different questions:

| | A. From the committed results | B. From scratch |
|---|---|---|
| Question it answers | do the tables and the text really follow from the runs? | do the runs themselves come out the same? |
| What you run | one script, `thesis_rebuild_from_results.sh` | every profile again, then A |
| Cost | about 5 minutes, plus about 50 min to re-check the calibration (§ 5) | more than 6 400 CPU-hours for the four per-fold phases alone: a cluster (§ 6.7) |
| What to expect | **byte-identical** tables, numbers and profiles | numbers close to, but not bit-identical with, the thesis, unless you use the same machine, build and thread count (§ 6.8) |
| Needs | the two bases, Python, LaTeX | the same, plus the C++ build and a Slurm cluster |

Start with A. It is cheap, it checks the whole second half of the chain, and B ends with it
anyway.

**What was run for this page.** Path A (§ 5) was run as written on 2026-11-26. Path B (§ 6)
quotes the commands and measured costs of the runs of record (2026-11-12 to 2026-11-26); it was
not run again for this page, which would take months of CPU.

## 2. Prerequisites (both paths)

| Need | How | Checked by |
|---|---|---|
| this repository | `git clone https://github.com/ensismoebius/doutorado` | — |
| Python 3, **one** shared env at `~/.venv` | the CMake configure step creates or updates it from `scripts/requirements.txt` (numpy, scipy, torch CPU, praat-parselmouth, …); never create a second env | `~/.venv/bin/python -c "import numpy, scipy, torch, parselmouth"` |
| `python3` of the system with numpy and scipy | the `*_significance.sh` scripts call `python3` | `python3 -c "import scipy"` |
| LaTeX | `latexmk`, `pdflatex`, `bibtex`, the `abntex2` class | `latexmk -v` |
| a C++20 toolchain, CMake ≥ 3.28, Ninja (path B only) | [Build System](../Guides/Build-System.md) | `cmake --preset=max-performance` succeeds |

Configure once (this also fills `~/.venv`):

```bash
cmake --preset=max-performance
```

## 3. The two bases

The thesis reads two SQLite files that are **not in git** (gigabytes, and their licences ask
you to get them from the source). They live in one directory, written `<databases>`:

```
<databases> = ~/Documentos/academico/UNESP/doutorado/databases
  database.sqlite        Coretto et al. 2017: 15 speakers, voice + EEG, pronounced and imagined speech
  biomex.sqlite          BIOMEX-DB 2021: 45 volunteers, pronounced digits (the replication)
  *.mirror-verified, *.verified     the verification stamps the thesis quotes
```

Build and verify them with [Data Bases](../Guides/Data-Bases.md). In short:

```bash
python3 scripts/data/import_mat_dataset_to_sqlite.py <folder with S01..S15> <databases>/database.sqlite
python3 scripts/data/coretto_verify_mirror.py        # EEG against the Zenodo copy -> stamp
python3 scripts/data/biomex_download.py              # 51 archives, SHA-256 checked
python3 scripts/data/biomex_to_sqlite.py             # inclusion rule, filter, resampling, pairing
python3 scripts/data/biomex_verify.py                # second derivation of every pair -> stamp
```

**Failure mode: loud.** `thesis_base_facts.py` (step 8 of § 5) refuses a stamp whose SHA-256 is
not the current file's. A base converted again without verifying stops the chain there.

## 4. The map: what each phase is

The thesis runs one C++ program, `thesis`, once per **profile**: a JSON file under
`src/experiments/thesis/profiles/` that declares every setting (nothing has a default). The
profiles are grouped in phases; each phase writes `results/thesis/<phase>/`.

| Phase (folder) | Profiles | What it answers | Thesis |
|---|---|---|---|
| `phase00` | 275 | which feature extractor separates speakers best, per signal (voice, EEG pronounced, EEG imagined), by the paraconsistent score $D_{pen}$; no classifier | ch. 9, Fase 00 |
| `perfold` | 100 | **the primary result**: authentication on Coretto, with the extractor and the classifier's hyperparameters chosen inside each outer fold | ch. 9, Fase 01 |
| `pfbiomex` | 48 | the same method, frozen, on BIOMEX-DB (confirmatory replication) | ch. 9, BIOMEX-DB |
| `pfworld`, `pfbxworld` | 150, 90 | per-fold EER under the WORLD simulated dysphonias, on each base | ch. 9, WORLD |
| `phase01` | 142 | the fixed configuration (Phase 00 winner, one network): the comparison, and the CV-scheme and standardization ablations | ch. 9, comparison |
| `biomex` | 48 | the fixed configuration on BIOMEX-DB | ch. 9 |
| `pilot` | 6 | does WORLD resynthesis alone change the speaker? | ch. 8 |
| `world`, `bxworld` | 150, 90 | the WORLD grid under the fixed configuration (margins and EER) | ch. 9, WORLD |

Why two protocols, and the pre-registered rules: [Thesis per-fold protocol](../Experiments/Thesis-Per-Fold-Protocol.md)
and [Experiment05](../Experiments/Thesis.md).

## 5. Path A — rebuild everything from the committed results

### 5.1 One command

```bash
scripts/pipeline/thesis/thesis_rebuild_from_results.sh --check
```

It runs the steps below in order, then compares every rebuilt file with git. The last lines
should read:

```
=== [11] every rebuilt file against git
software/nn/results/thesis/world_analysis/world_floor.json: equal within rel-tol 1e-12
REPRODUCED: every rebuilt profile, result-derived file and table is the committed result
```

| Option | Effect |
|---|---|
| `--check` | exit status 1 when a file is a different result (for scripts and CI) |
| `--no-calibration` | skip step 1, the only slow one; the committed calibration is then used unchecked, and the script says so |
| `--no-pdf` | stop before LaTeX |

The script trains nothing.

**Measured** on the author's machine (2026-11-26, 12 cores): about 5 minutes for steps 2–11,
and 48 min for the calibration (step 1) with the machine otherwise idle (1 h 47 min while it
was busy with other work; the run of record took 48 min too). Result: `REPRODUCED`.

**What "the same" means.** Most rebuilt files are byte-identical. Two kinds of difference are
not a different result, and `thesis_compare_rebuild.py` tells them apart from one that is:

| Difference | Example | Treated as |
|---|---|---|
| a wall-clock field | the calibration's `"seconds": 2900.9` (run of record) vs `6402.9` (rebuild) | not a result: ignored. Step 1 therefore writes into a scratch folder and only *compares*; the chain keeps the committed `calibration.json`, whose duration the thesis quotes (`cal.minutes`) |
| the last bits of a float | `world_floor.json`: `"r": 0.7091564366639` vs `0.7091564366639003` (relative 4e-16), from another summation order (numpy version, CPU) | equal when every number of the JSON file is within a relative 1e-12; the script prints the `git checkout` that restores the committed bytes |
| anything else | another key, another count, `0.7091` vs `0.7091564366639` | **a different result**: `NOT REPRODUCED` |

A non-JSON file (CSV, `.tex`) must be byte-identical. The PDF is left out of the comparison:
pdflatex writes the build date into it, so it differs at every build. Compare its text, or the
`thesis_numbers.tex` it is typeset from.

### 5.2 What each step does

Each step reads what an earlier one wrote. The table is the same order as the script.

| # | Script | Reads | Writes | Feeds |
|---|---|---|---|---|
| 1 | `scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py` | `database.sqlite` (vowels of S10, S03) | a scratch folder, compared with `results/thesis/dysphonia_calibration/{calibration.json,measurements.csv}` (in a run from scratch, § 6.4, it writes those) | the WORLD levels in the profiles (step 3); ch. 8 |
| 2 | `scripts/pipeline/thesis/01_thesis_phase00_rank.py` | `results/thesis/phase00/` | `winners.json`, `winner_stability.csv` | step 3; ch. 9 Fase 00 |
| 3 | `scripts/pipeline/thesis/00_thesis_generate_profiles.py` | `winners.json`, `calibration.json` | every profile under `src/experiments/thesis/profiles/`, `profiles.manifest` | the runs of § 6 |
| 4 | `thesis_build_phase00_paraconsistent_tables.py`, `thesis_build_phase01_auth_tables.py --protocol fixed`, `thesis_build_biomex_tables.py --protocol fixed` | `results/thesis/{phase00,phase01,biomex}/` | `tables/phase00_*.csv`, `phase01_*.csv`, `biomex_*.csv` | ch. 9 tables |
| 5 | the same two scripts with `--protocol per-fold` | `results/thesis/{perfold,pfbiomex}/` (+ the fixed results to compare) | `tables/perfold_*.csv`, `pfbiomex_*.csv` | ch. 9 tables |
| 6 | `03_thesis_phase01_significance.sh`, `04_thesis_biomex_significance.sh`, `10_thesis_perfold_significance.sh`, `11_thesis_pfbiomex_significance.sh` (all call `thesis_cross_profile_significance.py`) | per-fold metrics of each declared family | `results/thesis/<phase>/significance_*.csv` | the p-values of ch. 9 |
| 7 | `06_thesis_identity_pilot.py`, `07_thesis_identity_margin.py`, `08_thesis_world_analysis.py` (`--protocol fixed`, then `per-fold`), `09_thesis_world_floor.py` | `results/thesis/{pilot,identity_margin,world_margin,world,bxworld,pfworld,pfbxworld}/` | `pilot_verdict.json`, `identity_margin_verdict.json`, `results/thesis/world_analysis/*` | ch. 8 and ch. 9 WORLD |
| 8 | `scripts/analysis/thesis_base_facts.py`, `thesis_ae_input_reference.py`, `thesis_eeg_band_power.py` | the two bases and their stamps | `results/thesis/base_facts.csv`, `ae_input_reference.csv`, `eeg_band_power.csv` | ch. 2, ch. 5, ch. 8 |
| 9 | `scripts/pipeline/thesis/thesis_build_text_numbers.py` | everything above | `tables/thesis_numbers.tex` (about 6 500 keys) | every `\tn{key}` of the prose |
| 10 | `latexmk` in `documentation/00-thesis/monography/` | the chapters, tables, figures | `monografia.pdf` | — |
| 11 | `scripts/pipeline/thesis/thesis_compare_rebuild.py`, once per changed file | the rebuilt file and its committed version (`git show HEAD:<path>`) | a verdict per file | `REPRODUCED` / `NOT REPRODUCED` |

Step 1 is the slow one (about 50 minutes) and only re-checks a committed file; `--no-calibration`
skips it.

Step 9 is also a test of the text. Besides the numbers it checks 83 qualitative sentences of the
thesis (`CLAIMS` in the script: "the same extractor wins on the three signals", "the EEG beats the
voice in every standardized cell", …) against the data, and **stops** with the sentence if one
has become false. A number typed by hand into the text would compile and go stale silently;
that is why the prose has none ([Thesis Document](../Guides/Thesis-Document.md)).

### 5.3 Where each table of chapter 9 comes from

| Thesis table (label in `09-testsAndResults.tex`) | CSV | Step |
|---|---|---|
| Phase 00 rankings, one per family and signal | `phase00_{hc,ae,pca}_{voice,eegpron,eegimag}.csv` | 4 |
| per-fold authentication (`tab:perfoldAuth`) | `perfold_auth.csv` | 5 |
| what the folds chose (`tab:perfoldSelection`) | `perfold_selection.csv` | 5 |
| controls: DSNN vs its non-spiking twin vs cosine (`tab:perfoldControls`) | `perfold_controls.csv` | 5 |
| inference energy (`tab:perfoldEnergy`) | `perfold_energy.csv` | 5 |
| simulated dysphonia (`tab:perfoldDysphonia`) | `perfold_dysphonia.csv` | 5 |
| extractors compared under one classifier (`tab:perfoldExtractors`) | `perfold_extractors.csv` | 5 |
| fixed configuration and its ablations (`tab:phase01auth`) | `phase01_auth.csv` | 4 |
| per-fold vs fixed (`tab:perfoldVsFixed`) | `perfold_vs_fixed.csv` | 5 |
| BIOMEX-DB authentication, dysphonia, choices (`tab:pfbiomexAuth`, `tab:pfbiomexDysphonia`, `tab:pfbiomexSelection`) | `pfbiomex_{auth,dysphonia,selection}.csv` | 5 |

`phase01_{controls,dysphonia,energy,extractors}.csv` and `biomex_{auth,dysphonia}.csv` are
written too (the fixed configuration's versions); the current text quotes their numbers through
`thesis_numbers.tex` and typesets only `phase01_auth.csv`. The three hand-made `.tex` tables in
`tables/` (wavelet examples) hold no result.

## 6. Path B — run the experiments again

This is the expensive half. Read it to the end before starting anything: the order is forced by
the data, and most phases only make sense on a cluster.

### 6.1 Build and test

```bash
cmake --build out/build/max-performance --target thesis thesis_features -j$(nproc)
ctest --test-dir out/build/max-performance --output-on-failure -j4
~/.venv/bin/python scripts/testing/test_thesis_phase_scripts.py     # the Python side
```

`thesis --config <profile> [--repeat k]` runs one profile (or one repetition `k`); it refuses
any other argument. `thesis_features --config <profile> --out-dir <dir>` writes the clean and
degraded feature vector of every sample, with no classifier (steps 6.5 and 6.6).

### 6.2 Smoke test (minutes)

Every profile has a tiny mirror under `profiles/smoke/` (60 to 240 samples, 2 epochs, 2 folds,
one grid point). It runs every code path; its numbers mean nothing.

```bash
./scripts/testing/run_thesis_smoke.sh phase00
./scripts/testing/run_thesis_smoke.sh phase01
./scripts/testing/run_thesis_smoke.sh biomex
rm -rf results/thesis/smoke
```

### 6.3 The order

```
 a calibration ─┐
                ├─► b Phase 00 ─► rank ─► regenerate profiles ─┬─► c Phase 01 (fixed)
 (needs only    │                                              ├─► d BIOMEX-DB (fixed)
  database.sqlite)                                             ├─► e pilot, identity margin
                                                               └─► f WORLD grid (fixed)
 g per-fold protocol (needs no winners): warm cache ─► perfold ─► pfbiomex ─► pfworld ─► pfbxworld
 h Path A (§ 5)
```

The fixed-configuration profiles (c–f) carry the Phase 00 winners and the calibrated WORLD
requests, so they are valid only after a and b. A different machine may pick a different
winner in b; step 3 of § 5 then writes different c–f profiles, and the committed results no
longer match them. That is expected: those are new results.

### 6.4 a + b: calibration and Phase 00

```bash
~/.venv/bin/python scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py \
  --db ~/Documentos/academico/UNESP/doutorado/databases/database.sqlite \
  --subjects S10 S03 --jobs $(nproc) --out-dir results/thesis/dysphonia_calibration
./scripts/testing/run_thesis_profiles.sh phase00          # 275 profiles; locally feasible (hours)
```

On a pass with no failure the runner itself ranks the results, regenerates the profiles with
the winners and the calibration, and writes the Phase 00 tables (`post_process_phase00()` in
`run_thesis_profiles.sh`; any failure skips this, so a ranking never sees an incomplete set).
Review `git diff src/experiments/thesis/profiles` afterwards: the winners are now in the
profiles.

### 6.5 c–e: Phase 01, BIOMEX-DB, pilot and identity margin

```bash
./scripts/testing/run_thesis_profiles.sh phase01          # 142 profiles x 3 repetitions
./scripts/testing/run_thesis_profiles.sh biomex           # 48 profiles x 3
BIN=out/build/max-performance/src/experiments/thesis
for f in src/experiments/thesis/profiles/pilot/*.json; do $BIN/thesis --config "$f"; done   # ~42 min
for p in pilot/pil_cosine_voice_control_probes phase01/p01_dysph_voice_probes_l25; do
  $BIN/thesis_features --config src/experiments/thesis/profiles/$p.json --out-dir results/thesis/identity_margin
done
```

Phase 01 and BIOMEX-DB run locally too, but are days of CPU; on the cluster each is one Slurm
array (§ 6.7).

### 6.6 f: the WORLD grid (fixed configuration)

```bash
for f in src/experiments/thesis/profiles/world/p01w_voice_probes_*.json \
         src/experiments/thesis/profiles/bxworld/bx01w_voice_probes_*.json; do
  $BIN/thesis_features --config "$f" --out-dir results/thesis/world_margin
done
```

and, on the cluster, the 150 `world` and 90 `bxworld` profiles (§ 6.7).

### 6.7 On the cluster (GridUnesp, Slurm)

One phase = one array of `scripts/pipeline/thesis/thesis_gridunesp.sbatch`; each task asks for
1 CPU, 16 GB and 1 day on the `short` partition (the warm array: 1 CPU, 8 GB, 8 h).
`scripts/pipeline/meeting01/gridunesp_deploy.sh` copies the tree, builds on a compute node and
sends the bases ([GridUnesp Deployment](../Guides/GridUnesp-Deployment.md)).

| Phase | Submit (from `~/software/nn` on the cluster) | Measured cost |
|---|---|---|
| `phase00` | `THESIS_PHASE=phase00 sbatch --array=0-63 scripts/pipeline/thesis/thesis_gridunesp.sbatch` | slowest profile 230 s |
| `phase01` | `THESIS_PHASE=phase01 sbatch --array=0-141 …` | not recorded as a total |
| `biomex` | `THESIS_PHASE=biomex sbatch --array=0-47 …` | not recorded as a total |
| `world`, `bxworld` | `THESIS_PHASE=world sbatch --array=0-149 …`, `THESIS_PHASE=bxworld sbatch --array=0-89 …` | not recorded as a total |
| feature cache (before any per-fold phase, once per deploy) | `sbatch --array=0-10 scripts/pipeline/thesis/thesis_warm_feature_cache.sbatch` | 8 min to 3 h 47 min per task |
| `perfold` | `THESIS_PHASE=perfold sbatch --array=0-99 …` | ~710 CPU·h |
| `pfbiomex` | `THESIS_PHASE=pfbiomex sbatch --array=0-47 …` | ~1 036 CPU·h, peak 9.3 GB |
| `pfworld` | `THESIS_PHASE=pfworld sbatch --array=0-149 …` | ~1 614 CPU·h |
| `pfbxworld` | `THESIS_PHASE=pfbxworld sbatch --array=0-269 …` | ~3 063 CPU·h (one unit per task: ~11 h, max 13.9 h) |

Watch an array with `scripts/pipeline/thesis/thesis_grid_watch.sh` (one ssh session; repeated
logins trip the cluster's Fail2Ban). When an array is done, bring its folder home:

```bash
rsync -avz <user>@access.grid.unesp.br:software/nn/results/thesis/<phase>/ results/thesis/<phase>/
```

Each finished unit writes `.done/<unit>.done` with the code revision and thread count; a phase
folder that holds markers of another revision or thread count is refused, so two builds never
mix in one result set. Every refusal is listed in
[Running Thesis Profiles § 10](../Guides/Running-Thesis-Profiles.md#10-per-fold-protocol--extractor-and-hyperparameters-chosen-inside-each-fold)
and [Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12).

### 6.8 How close your numbers will be

**The same profile, build, CPU and thread count give byte-identical output** (checked on
2026-11-12: a repetition run alone and inside the full run matched in every file except the
wall-clock fields). **A different machine does not.** At one thread the same 3-epoch profile gave
EER 0.4176 on a cluster node and 0.4239 on the author's desktop: another CPU, `-march=native`,
another BLAS, and training amplifies the difference in summation order. The thread count changes
results the same way (0.4176 / 0.4083 / 0.4201 at 1 / 2 / 4 threads), which is why every run of
record used one thread.

So a run from scratch on other hardware reproduces the *method* and should reproduce the
*conclusions*; it will not reproduce every fourth decimal. When it does not, step 9 of § 5 tells
you which sentence of the thesis no longer holds.

The handcrafted extractors train nothing, so their Phase 00 scores are the exception: they
should come back bit-identical anywhere. If they move, stop and investigate.

### 6.9 h: back to Path A

When every phase is home, run § 5. Commit the results, the regenerated profiles, the tables
and `thesis_numbers.tex` together, so the text never quotes a number its results do not hold.

## 7. Every file of the result tree

| Folder or file under `results/thesis/` | Written by | Read by |
|---|---|---|
| `phase00/` (+ `winners.json`, `winner_stability.csv`) | `thesis` on `profiles/phase00`; `01_thesis_phase00_rank.py` | `00_`, the Phase 00 tables, the text numbers |
| `phase01/`, `biomex/` (+ `significance_*.csv`) | `thesis` on `profiles/{phase01,biomex}`; `03_`/`04_` | the fixed tables, the text numbers |
| `biomex/coretto_phase00_winners_frozen.json` | committed by hand before any BIOMEX-DB result existed | nobody: it is the record that the BIOMEX-DB profiles carry the Coretto winners, chosen before the replication |
| `perfold/`, `pfbiomex/` (+ `significance_*.csv`) | `thesis` on the per-fold profiles; `10_`/`11_` | the per-fold tables, the text numbers |
| `pilot/` (+ `pilot_verdict.json`) | `thesis` on `profiles/pilot`; `06_` | the text numbers (ch. 8) |
| `identity_margin/` | `thesis_features`; `07_` | the text numbers (ch. 8) |
| `world/`, `bxworld/`, `pfworld/`, `pfbxworld/` | `thesis` on the WORLD profiles | `08_` (Part B, EER) |
| `world_margin/` | `thesis_features` on the 30 `voice_probes` WORLD profiles | `08_` (Part A, margins), `09_` |
| `world_analysis/` | `08_`, `09_` | the text numbers (ch. 9) |
| `dysphonia_calibration/` | `05_` | `00_` (WORLD levels), the text numbers |
| `base_facts.csv`, `ae_input_reference.csv`, `eeg_band_power.csv` | `scripts/analysis/thesis_*.py` | the text numbers |

Inside a phase folder every run writes `e05_<profile>_rep<k>_*`: `summary.json` (the
configuration it used and every metric), `metrics.csv` (one row per fold), `paraconsistent.csv`,
`learning_curves.dat`, and for the per-fold protocol `selection_extractors.csv` and
`selection_grid.csv` (what each fold chose). The trained model of every outer fold is under
`models/`. Full list: [Running Thesis Profiles › Per-run diagnostics](../Guides/Running-Thesis-Profiles.md#per-run-diagnostics-output-files).

## 8. Every script

Pipeline, in the order of use (`scripts/pipeline/thesis/`):

| Script | One line |
|---|---|
| `00_thesis_generate_profiles.py` | writes every profile and `profiles.manifest`; `--phase00-only` before the ranking, `--winners … --fused voice --calibration …` after it. Never edit a profile by hand: CI rebuilds the tree in memory and refuses a difference |
| `01_thesis_phase00_rank.py` | ranks Phase 00 by $D_{pen}$, writes the winner per signal and the best per family; refuses an incomplete set, reports exact ties |
| `03_`, `04_`, `10_`, `11_…_significance.sh` | the significance tests declared before the results, per family, Holm within the family; one script per result set (fixed Coretto, fixed BIOMEX-DB, per-fold Coretto, per-fold BIOMEX-DB) |
| `05_thesis_dysphonia_calibration.py` | finds, with Praat on real vowels, the WORLD request of each rough/tremor level that yields its measured target |
| `06_thesis_identity_pilot.py` | verdict of the identity pilot (does resynthesis alone move the EER?) |
| `07_thesis_identity_margin.py` | the powered identity test on per-utterance margins |
| `08_thesis_world_analysis.py` | the WORLD grid: Part A margins (confirmatory, fixed only) and Part B EER (both protocols) |
| `09_thesis_world_floor.py` | the pre-registered floor test of the BIOMEX-DB WORLD null |
| `thesis_cross_profile_significance.py` | the test engine the `.sh` scripts call |
| `thesis_build_phase00_paraconsistent_tables.py`, `thesis_build_phase01_auth_tables.py`, `thesis_build_biomex_tables.py` | the ch. 9 tables (§ 5.3) |
| `thesis_build_text_numbers.py` | `thesis_numbers.tex` and the `CLAIMS` check |
| `thesis_rebuild_from_results.sh` | Path A in one command (§ 5) |
| `thesis_compare_rebuild.py` | is a rebuilt file the committed result? byte-identical, or JSON within a stated relative tolerance, with named wall-clock keys ignored |
| `thesis_gridunesp.sbatch`, `thesis_warm_feature_cache.sbatch` | one phase as a Slurm array; the feature cache before the per-fold phases |
| `thesis_grid_watch.sh`, `thesis_grid_status.py` | live view of the arrays on the cluster |

Elsewhere:

| Script | One line |
|---|---|
| `scripts/testing/run_thesis_profiles.sh`, `run_thesis_smoke.sh` | run a phase locally (real or smoke profiles), resumable, memory-gated parallelism |
| `scripts/testing/test_thesis_phase_scripts.py`, `test_thesis_base_checks.py` | the tests of the Python side (CI runs them) |
| `scripts/analysis/thesis_base_facts.py`, `thesis_ae_input_reference.py`, `thesis_eeg_band_power.py` | facts about the bases, the AE-input reference, the EEG band power |
| `scripts/data/*.py` | build and verify the bases (§ 3) |
| `scripts/lib/thesis_dysphonia_lib.py` | compiles the thesis's own C++ dysphonia code so Python calls the same code |
| `documentation/00-thesis/monography/images/generate_*.py` | the three generated figures ([Thesis Document › Generated figures](../Guides/Thesis-Document.md#generated-figures)) |

## 9. When something goes wrong

| Symptom | Meaning | Loud or silent |
|---|---|---|
| `REFUSED: … is missing` from the rebuild script | a base or `~/.venv` is not where it should be | loud |
| a script names a missing results file | a phase is incomplete; every script refuses a partial set rather than averaging what exists | loud |
| `N statement(s) of the thesis prose no longer hold` (step 9) | the data contradicts a sentence of the thesis: rewrite the sentence, never loosen the claim | loud |
| `thesis_base_facts.py` refuses a stamp | the base changed after it was verified | loud |
| `NOT REPRODUCED` with changed profiles | your Phase 00 picked other winners (other machine), or the generator changed | loud (exit 1 with `--check`) |
| `NOT REPRODUCED` on a result file | the committed result and its rebuild differ beyond rounding: a script changed what it computes | loud (exit 1 with `--check`) |
| `equal within rel-tol 1e-12` | float summation order (numpy, CPU); no printed number moves | reported, not a failure |
| numbers differ in the third or fourth decimal after a run from scratch | another CPU, build or thread count (§ 6.8) | silent, and expected |

## Related

- [Running Thesis Profiles](../Guides/Running-Thesis-Profiles.md) — every runner option, the caches, the outputs
- [Thesis Document](../Guides/Thesis-Document.md) — how the text is built from the results; adding a number
- [Data Bases](../Guides/Data-Bases.md) — obtaining, converting and verifying the bases
- [Experiment05](../Experiments/Thesis.md) — the design, its history, every defect found and fixed
- [Thesis per-fold protocol](../Experiments/Thesis-Per-Fold-Protocol.md) — the pre-registration of the primary result
