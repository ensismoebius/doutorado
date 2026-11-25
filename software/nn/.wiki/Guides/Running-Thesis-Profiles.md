# Running Thesis Profiles (Experiment05)

How to run the thesis pipeline end-to-end: Phase 00 (feature-vector construction +
paraconsistent ranking) → Phase 01 (DSNN authentication) → the BIOMEX-DB replication. All
commands run from `software/nn/`. What happens to the results afterwards (tables, tests,
numbers in the thesis text) is in [Thesis Document](./Thesis-Document.md). See [Experiment05](../Experiments/Thesis.md)
for the design.

## Prerequisites

- The bases at the paths the profiles name: `~/Documentos/academico/UNESP/doutorado/databases/database.sqlite` (Coretto; Phase 00/01) and
  `~/Documentos/academico/UNESP/doutorado/databases/biomex.sqlite` (BIOMEX-DB). How to build both: [Data Bases](./Data-Bases.md).
- The `thesis` binary built in any CMake profile:

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target thesis -j$(nproc)
```

## Runner scripts

Two runners share one live-progress UI, failure capture, and binary auto-detection:

| Script | Runs | Params |
|---|---|---|
| `scripts/testing/run_thesis_profiles.sh [phase00\|phase01\|biomex\|all\|start-over-all]` | the **real** profiles | full (the actual experiment) |
| `scripts/testing/run_thesis_smoke.sh [phase00\|phase01\|biomex\|all]` | the **smoke** mirror | tiny (fast crash check) |

`run_thesis_profiles.sh` with no argument opens a menu on a terminal and **refuses** in a pipe
(there is no default scope: the old default, `all`, trains Phase 01 on the previous winners).

**Binary selection** (works with any build profile):

- auto: the most recently built `out/build/*/…/thesis`;
- `THESIS_BUILD=max-performance-device ./scripts/testing/run_thesis_profiles.sh phase00`;
- `THESIS_BIN=/abs/path/to/thesis ./scripts/testing/run_thesis_profiles.sh`.

**Progress**: before each profile the runner prints
`[i/N] elapsed h:mm:ss eta~h:mm:ss running: <profile>`. On a terminal the
binary's own live progress bars then render underneath — **dataset loading,
feature extraction, training epochs, and CV folds** (driven by `ProgressManager`
/ `ProgressCallback`) — while output is still captured for failure diagnosis. In
a pipe/CI the ANSI bars are redirected to keep logs clean. Failures print
immediately and are listed in the final summary.

**Rich per-bar metadata (Meeting01/Meeting01 look)**: each training bar carries a
metadata line — `label │ description │ run X/Y  LOSS │ phases`. The autoencoder
feature-extraction bars name the model + loss (`SNN-AE (poisson)`, `ANN-AE
(direct)`, `LSTM-AE` │ `MSE`); the DSNN classifier bars show `run fold/total`
and `CrossEntropy` with live train/val loss. This is the same
`ProgressCallback::set_metadata` path Meeting01 uses, so both experiments read alike.

**Overall run banner + work-weighted ETA**: each profile is a *separate process*
and cannot know the whole-run progress, so the runner computes an overall status
and passes it to the binary via `THESIS_OVERALL` — rendered as a persistent top
line (`Overall [i/N] elapsed … ETA …`) above the profile's own bars, mirroring
Meeting01's `MEETING01_OVERALL`. The overall ETA is **work-weighted + EMA-smoothed**
(`scripts/lib/run_eta.sh`): each profile is weighted by rough cost
(`p00_hc_*` = 1, autoencoders / DSNN = 20) and the runner tracks
seconds-per-unit-work rather than counting profiles equally. This matters because
the mix is wildly heterogeneous — a naive per-profile mean reads ~7× too high
through the heavy-AE phase and only corrects near the end; the weighted estimate
stays stable from the first completed AE. In the parallel monitor the same
weighting drives the dashboard's `overall ETA` (workers append their weight to
`$TMP/weights_done`; the monitor sums it each frame).

## Live dashboard (parallel runs)

With `JOBS>1` on a terminal (`THESIS_MONITOR` unset or `1`), the workers' own
progress bars can't share one screen, so a background `monitor_loop()`
redraws a full dashboard in place every `THESIS_MONITOR_INTERVAL` seconds
(default 2s):

- **Header** — work-weighted progress bar and `%`, `done/total` profiles,
  pass/fail/skip counts, running-worker count, elapsed time, ETA, projected
  wall-clock finish time, throughput (profiles/hour), and host pressure
  (`jobs/cpus`, RAM used/total, swap used, load average, active build).
- **Worker panels** — one 4-row block per running profile: profile name +
  per-worker runtime; modality/strategy/dataset shape and the current
  `repeat i/n seed=s`; then the two live bars (outer = epochs/folds, inner =
  batches) each redrawn as a small `[##--------]` bar with pct, counts, loss,
  and per-bar ETA. Static metadata (modality/strategy/dataset shape) is
  parsed once from each log's head and cached; live state (repeat/seed,
  current bars) is re-parsed from a bounded tail every frame, so redraw cost
  doesn't grow with run length.
- **Recent panel** — the last 3 finished profiles with PASS/FAIL and wall
  time, shown only if the terminal has spare rows.

Every frame is clamped to the terminal's actual size, read via
`stty size < /dev/tty` rather than `tput lines/cols`: `tput`'s size ioctl
fails silently in the background subshell the monitor runs in (stdin is
`/dev/null` there), falling back to terminfo's static 80×24 default — which
used to clamp every frame to 80 columns and mis-budget the height regardless
of the real terminal size. `THESIS_MONITOR=0` disables the dashboard in favor of
plain `start:`/`PASS`/`FAIL` event lines; `THESIS_MONITOR_TAIL_BYTES` /
`THESIS_MONITOR_HEAD_BYTES` bound how much of each (multi-MB) worker log is
re-read per frame.

Serial runs (`JOBS=1`) skip the dashboard entirely — the binary's own bars
render directly, as described above.

## Parallelism (memory-gated)

`run_thesis_profiles.sh` runs several profiles at once — each is an independent
process with its own `run_tag`, so phase00/phase01 parallelise safely as far
as *correctness* goes. Job count is computed as:

```
JOBS = min(MemAvailable / THESIS_JOB_MEM_MB, nproc), capped at 4
```

Override with `THESIS_JOBS` directly, or tune the per-job budget with
`THESIS_JOB_MEM_MB` (default **5120**).

**2026-07-14 fix**: the default used to be 2048MB. Measured peak RSS+swap for
real phase00 profiles: `snn-ae`/poisson **voice** profiles plateau around
**4.4GB**, EEG profiles around **2.1GB** (`kAeInputFeatures=256`,
`time_steps=16`, ~1974 voice samples → 31,584 spike frames trained for
`epochs=100`). With the old 2048MB budget the script would happily launch 4
voice jobs in parallel on a machine that could only actually hold 1–2,
pushing the box into heavy swap (14GB+ swap used on a 17GB machine). The
budget is now 5120MB so `JOBS` auto-throttles correctly; if you profile a
config with a different memory footprint, pass `THESIS_JOB_MEM_MB` explicitly
rather than trusting the default blindly.

The 2GB-per-job assumption wasn't a leak in `thesis` itself — see
[Memory Diagnostics](./Memory-Diagnostics.md) for the full
investigation writeup and the general leak-vs-plateau method.

**2026-07-15 fixes** (both in `run_thesis_profiles.sh`):
- **Throttle correctness.** The parallel pool used to gate on `$TMP/active`
  marker files. A worker stopped by `SIGTTOU`/`SIGTTIN` (the script
  backgrounded while something touches the TTY) never removes its marker, and
  a signal-interrupted `wait -n` returns without reaping — so the gate spun
  through and launched **every** pending profile at once (observed twice:
  ~30 concurrent `thesis` processes, OOM-killing unrelated work). It now
  throttles on live worker PIDs, which is immune to both.
- **Binary auto-pick.** Auto-pick now prefers the `max-performance` (CPU) build
  when present, instead of "most recently built". These profiles' networks are
  tiny, so a stray rebuild of another backend preset used to silently switch
  runs onto a slower binary. Override with `THESIS_BUILD` / `THESIS_BIN` to
  target a specific backend.

## Crash / power-loss recovery

`run_thesis_profiles.sh` checkpoints every completed profile to
`results/thesis/run_profiles_<scope>.state`. Because the binary writes each profile's
result files (CSV/JSON) to disk as it finishes, a power loss only costs the
profile that was mid-run — it isn't recorded and re-runs on restart.

On start, if a state file exists the runner reports how many profiles are already
done and asks:

```
Found a previous run: 147 profile(s) already completed → results/thesis/run_profiles_phase00.state
Resume (skip completed) [R], start over [S], abort [A]?
```

- **Resume** skips the completed profiles and continues.
- **Start over** clears the state and re-runs everything.
- Non-interactive (nohup/CI) defaults to **resume**; force with `RESUME=1` or
  `FRESH=1` (start over). Delete the `.state` file to force a clean run.

## Smoke first (recommended)

Confirm every code path runs before committing hours to the full sweep:

```bash
./scripts/testing/run_thesis_smoke.sh phase00
./scripts/testing/run_thesis_smoke.sh phase01
```

The smoke mirror (`profiles/smoke/`) is auto-regenerated by the CMake
`thesis_smoke_profiles` target whenever a real profile changes. See
[Ground-Truth and Smoke Testing](./Ground-Truth-and-Smoke-Testing.md).

## Full pipeline

The order is forced by the data: Phase 01 and BIOMEX-DB profiles carry the Phase 00 winners,
so they are only valid after Phase 00 ran and its winners were written in.

```
1 phase00 ──► (automatic, on a clean pass) 2 rank ─► 3 regenerate profiles ─► 4 Phase 00 tables
5 phase01 ──► 6 Phase 01 tables, significance
7 biomex  ──► 8 BIOMEX tables, significance
9 WORLD dysphonia: a calibration (before 3) ─► b pilot ─► c identity margin ─► d grid ─► e floor
10 per-fold protocol (cluster): warm the feature cache ─► perfold ─► pfbiomex ─► pfworld ─► pfbxworld
then: the thesis numbers (Thesis Document guide)
```

Step 10 needs no winners: each fold chooses its own extractor and hyperparameters. It is the
primary result from 2026-11-18; steps 1–9 are the fixed-configuration comparison.

`./scripts/testing/run_thesis_profiles.sh start-over-all` runs 1, 5 and 7 in that order from
scratch, stopping at the first leg that fails (2–4 run inside leg 1). Locally every profile runs
on this machine; on the cluster each phase is one Slurm array
([Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12)).

### 1. Phase 00 — feature ranking

Runs every profile in `profiles/phase00/` (275: handcrafted, ANN-AE, SNN-AE and PCA for each of
`voice`, `eegpron`, `eegimag`; `classifier.enabled=false`: features are scored, no classifier is
trained). Writes `results/thesis/phase00/e05_<tag>_paraconsistent.csv` and summaries.

```bash
nohup ./scripts/testing/run_thesis_profiles.sh phase00 > phase00_run.log 2>&1 &   # heavy
```

On a pass with **zero failures** the runner then runs steps 2–4 itself
(`post_process_phase00()`); they are listed below for a manual re-run.

- **Fails closed.** Any failed profile skips the post-processing: a ranking over an incomplete
  set could pick the wrong winner. `01_thesis_phase00_rank.py` and the table script also refuse
  an incomplete set themselves (every repetition a profile declares must exist).
- `THESIS_SKIP_POST=1` disables the post-processing.
- Step 3 **rewrites the tracked `phase01/` and `biomex/` profiles** — review
  `git diff src/experiments/thesis/profiles` before committing.

### 2. Rank → winner per signal

```bash
python3 scripts/pipeline/thesis/01_thesis_phase00_rank.py \
  --profiles-dir src/experiments/thesis/profiles/phase00 \
  --results-dir  results/thesis/phase00 \
  --out          results/thesis/phase00/winners.json
```

Selects on `d_penalized` (not raw `d_truth`, which a collapsed latent can game; see
[Paraconsistent](../Core/Paraconsistent.md)) and reports exact ties (`[TIE]`: the winner was
chosen by the tie-break, not by merit). It reads exactly the files each profile declares and
**refuses** when one is missing, holds more than one scored row, or lacks `d_penalized`. The
output also records the best profile of each family (`hc`, `ae`, `pca`), which the
extractor-comparison profiles of Phase 01 use.

### 3. Write the winners into the Phase 01 and BIOMEX-DB profiles

```bash
python3 scripts/pipeline/thesis/00_thesis_generate_profiles.py \
  --winners results/thesis/phase00/winners.json --fused voice \
  --calibration results/thesis/dysphonia_calibration/calibration.json
```

`--fused voice|eeg` (required) chooses which signal's winner backs the fused profiles; the run of
record uses `voice`. `--calibration` is the output of step 9a, which needs only the Coretto base,
so it can run before Phase 00. Single-signal profiles get their own signal's winner, and
`p01_cmp-<family>_<signal>_*` its family's best. The manifest records the winners file's sha256;
`GeneratedTree` in `scripts/testing/test_thesis_phase_scripts.py` (run by CI) rebuilds the tree
in memory and refuses any difference, so a stale tree fails CI loudly.
`git checkout src/experiments/thesis/profiles` restores the tracked tree if a smoke or partial
`winners.json` was used by mistake.

### 4. Phase 00 tables

```bash
python3 scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py \
  --profiles-dir src/experiments/thesis/profiles/phase00 \
  --results-dir  results/thesis/phase00 \
  --tables-dir   ../../documentation/00-thesis/monography/tables
```

Writes `tables/phase00_{ae,hc,pca}_{voice,eegpron,eegimag}.csv`, ranked **ascending** by
`d_penalized` (rank 1 = best; an earlier version sorted descending — see
[Experiment05 pitfall 12](../Experiments/Thesis.md#common-pitfalls)), the winner marked with a
dagger. A group the thesis typesets with no profile is refused rather than left stale.

### 5. Phase 01 — authentication

```bash
./scripts/testing/run_thesis_profiles.sh phase01
```

Runs `profiles/phase01/` (142: the source × text-mode × CV grid, the ANN/cosine controls, the
simulated dysphonia, the extractor comparison). Writes `results/thesis/phase01/`, including the
trained model of every outer fold under `models/`.

### 6. Phase 01 tables and significance

```bash
python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py --protocol fixed \
  --profiles-dir src/experiments/thesis/profiles/phase01 \
  --results-dir  results/thesis/phase01 \
  --tables-dir   ../../documentation/00-thesis/monography/tables
scripts/pipeline/thesis/03_thesis_phase01_significance.sh \
  src/experiments/thesis/profiles/phase01 results/thesis/phase01
```

The first writes `tables/phase01_{auth,controls,dysphonia,energy,extractors}.csv` (means over
the repetitions; every profile's summaries must exist and hold finite EER/AUC, else it refuses).
The second runs **only the tests declared in it** (families `vsvoice`, `fusion`, `ctrl`,
`dysph`, `cmp`, Holm within each family) and writes `results/thesis/phase01/significance_*.csv`.
A test chosen after looking at the table is not a test, so add a family there *before* reading
the results.

### 7. BIOMEX-DB replication

```bash
./scripts/testing/run_thesis_profiles.sh biomex
```

Runs `profiles/biomex/` (48: the Phase 01 cells on the second base, with the **Coretto** Phase 00
winners — nothing is selected on BIOMEX-DB). Needs `~/Documentos/academico/UNESP/doutorado/databases/biomex.sqlite`.

### 8. BIOMEX tables and significance

```bash
python3 scripts/pipeline/thesis/thesis_build_biomex_tables.py --protocol fixed \
  --profiles-dir src/experiments/thesis/profiles/biomex --results-dir results/thesis/biomex \
  --coretto-profiles-dir src/experiments/thesis/profiles/phase01 \
  --coretto-results-dir results/thesis/phase01 \
  --tables-dir ../../documentation/00-thesis/monography/tables
scripts/pipeline/thesis/04_thesis_biomex_significance.sh \
  src/experiments/thesis/profiles/biomex results/thesis/biomex
```

Writes `tables/biomex_{auth,dysphonia}.csv` (each BIOMEX cell beside its Coretto cell) and
`results/thesis/biomex/significance_*.csv` (the Coretto families, declared before reading).

### 9. WORLD dysphonia — calibration, identity tests, grid, floor

Why each step exists, and each pre-registered rule, is in
[Experiment05 › other dysphonia generators](../Experiments/Thesis.md#41-the-other-dysphonia-generators-degradationprofile-2026-11-15) and the
[proposal](../Experiments/Thesis-Voice-Degradation-Proposal.md) (§3.4 calibration, §4.3 pilot,
§4.4 powered test, §4.5 grid and floor). Every rule was committed **before** its run; a new test
gets its rule written into the script's docstring and committed first.

The feature runs below use the `thesis_features` binary (`--config <profile> --out-dir <dir>`,
both required): it writes the clean and degraded feature vector of every sample, one CSV row each,
through the same loader, extractor and degradation the `thesis` binary runs.

```bash
BIN=out/build/max-performance/src/experiments/thesis
cmake --build out/build/max-performance --target thesis thesis_features -j$(nproc)
```

**a. Calibration** (minutes, no training; Praat 6.1.38 through parselmouth). Finds, on real
vowels, the request of each rough/tremor level that yields the measured target.

```bash
~/.venv/bin/python scripts/pipeline/thesis/05_thesis_dysphonia_calibration.py \
  --db ~/Documentos/academico/UNESP/doutorado/databases/database.sqlite \
  --subjects S10 S03 --jobs 10 --out-dir results/thesis/dysphonia_calibration
```

**b. Identity pilot** (does resynthesis alone move the EER? 6 profiles, ~42 min locally):

```bash
for f in src/experiments/thesis/profiles/pilot/*.json; do $BIN/thesis --config "$f"; done
python3 scripts/pipeline/thesis/06_thesis_identity_pilot.py \
  --results results/thesis/pilot --out results/thesis/pilot/pilot_verdict.json
```

**c. Powered identity test** (per-utterance margins, no training):

```bash
for p in pilot/pil_cosine_voice_control_probes phase01/p01_dysph_voice_probes_l25; do
  $BIN/thesis_features --config src/experiments/thesis/profiles/$p.json \
    --out-dir results/thesis/identity_margin
done
python3 scripts/pipeline/thesis/07_thesis_identity_margin.py \
  --control results/thesis/identity_margin/e05_pil_cosine_voice_control_probes \
  --whisper results/thesis/identity_margin/e05_p01_dysph_voice_probes_l25 \
  --out results/thesis/identity_margin/identity_margin_verdict.json
```

**d. WORLD grid.** 150 Coretto profiles (`profiles/world/p01w_*`) and 90 BIOMEX-DB profiles
(`profiles/bxworld/bx01w_*`), every level read against the control of its own cell. Run on
GridUnesp, one whole profile per task (`THESIS_PHASE=world sbatch --array=0-149 …`,
`THESIS_PHASE=bxworld sbatch --array=0-89 …`;
[Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12)).
The confirmatory Part A reads features of the 15 `voice_probes` profiles of each base:

```bash
for f in src/experiments/thesis/profiles/world/p01w_voice_probes_*.json \
         src/experiments/thesis/profiles/bxworld/bx01w_voice_probes_*.json; do
  $BIN/thesis_features --config "$f" --out-dir results/thesis/world_margin
done
python3 scripts/pipeline/thesis/08_thesis_world_analysis.py --protocol fixed \
  --features results/thesis/world_margin \
  --profiles-root src/experiments/thesis/profiles --results-root results/thesis \
  --out results/thesis/world_analysis
```

Writes `world_margin.json` (Part A), `world_eer.csv` (Part B) and `sig_*.csv`.
Under the per-fold protocol only Part B is run (Part A reads `thesis_features`, one extractor per
profile and no classifier, so it does not depend on the protocol):

```bash
python3 scripts/pipeline/thesis/08_thesis_world_analysis.py --protocol per-fold \
  --bases coretto2017,biomex2021 \
  --profiles-root src/experiments/thesis/profiles --results-root results/thesis \
  --out results/thesis/world_analysis
```

It writes `world_eer_perfold.csv` and `sig_pfworld_*.csv` / `sig_pfbxworld_*.csv`. `--bases` is
required: name only `coretto2017` while `pfbxworld` is still running. It refuses when the
clean features of the runs of one base differ, or a profile's results are missing.

**e. Floor test** (is the BIOMEX-DB null a floor effect? cross-half correlation, 100000
permutations):

```bash
python3 scripts/pipeline/thesis/09_thesis_world_floor.py \
  --features results/thesis/world_margin --out results/thesis/world_analysis/world_floor.json
```

The number builder reads 9a–9d (`--dysphonia-calibration`, `--identity-pilot`,
`--identity-margin`, `--world-analysis`; [Thesis Document › Results to text](Thesis-Document.md#results-to-text)).

### 10. Per-fold protocol — extractor and hyperparameters chosen inside each fold

**Why it exists.** In steps 1–5 the extractor is chosen by $D_{\text{pen}}$ on all 15 speakers,
so the 3 test speakers of a fold helped choose the extractor they are then scored with. The
classifier is also never tuned. Here the choice moves inside the fold:

```
outer fold f ── 12 training speakers ──┬─ rank every Phase 00 candidate (D_pen on these rows)
                                       ├─ inner 4-fold search over the 36-point grid
                                       └─ refit the winner on all 12
             └─ 3 test speakers ──────── scored ONCE
```

The design, the rows of each split and the measured costs are in
[Thesis per-fold protocol](../Experiments/Thesis-Per-Fold-Protocol.md). This section only
covers how to run it.

**The profiles** are generated together with the rest (`00_thesis_generate_profiles.py`). They
carry a `selection` block and **no** `feature_extraction`, and they read no winners file.

| Phase (`THESIS_PHASE`) | Profiles | Units | Array | Results |
|---|---|---|---|---|
| `perfold` | 100 (`profiles/perfold/pf01_*`) | 100 × 3 repetitions | `--array=0-99` (3 units per task) | `results/thesis/perfold/` |
| `pfbiomex` | 48 (`pfbiomex/pfbx01_*`) | 48 × 3 | `--array=0-47` | `results/thesis/pfbiomex/` |
| `pfworld` | 150 (`pfworld/pf01w_*`) | 150 whole profiles | `--array=0-149` | `results/thesis/pfworld/` |
| `pfbxworld` | 90 (`pfbxworld/pfbx01w_*`) | 90 × 3 | `--array=0-269` (1 unit per task) | `results/thesis/pfbxworld/` |

`pfbxworld` is split by repetition because a whole BIOMEX-DB WORLD profile takes about 43 h,
which is more than the partition's one-day limit. For the same reason it runs ONE unit per task
(`--array=0-269`): a unit redoes the profile's WORLD resynthesis and takes ~11 h (mean 11.2 h,
max 12.5 h, measured on job 5345820; ~17 h was projected at submission), so three in a row
(`--array=0-89`) would pass the limit.

**Run on GridUnesp only, in this order.** These are cluster runs: a single voice unit costs
hours of CPU even with the cache (table below), and `run_thesis_profiles.sh` has no per-fold
scope.

```bash
# local
./scripts/pipeline/meeting01/gridunesp_deploy.sh
# cluster, from ~/software/nn
# 1. the feature cache, once per deploy (11 tasks, 8 min to 3 h 47 min each)
sbatch --array=0-10 scripts/pipeline/thesis/thesis_warm_feature_cache.sbatch
# 2. when every warm task is COMPLETED: one phase at a time (# CONFIRMED)
THESIS_PHASE=perfold sbatch --array=0-99 scripts/pipeline/thesis/thesis_gridunesp.sbatch
# 3. local, when the array is done: bring the phase home (no --delete)
rsync -avz <user>@access.grid.unesp.br:software/nn/results/thesis/perfold/ results/thesis/perfold/
```

**Watching a phase.** `scripts/pipeline/thesis/thesis_grid_watch.sh` (from `software/nn`) opens
one ssh session and refreshes in place every 60 s (`--interval`, `--once`, `--jobs`,
`--since-hours`). Per job it shows the phase, the tasks running and pending, the units done and
failed against the phase's total (the `.done` markers, the units of record), the same bar with
the running units' finished folds counted (it moves during day-long units), the mean unit time
and a time-left estimate. For each running task: its unit, the unit's finished outer folds
(`per-fold <f>:` lines of the unit's log over repetitions × `k_folds`), a progress bar of the
task (its finished units plus the current unit's fold share), how long the unit has run, the
measured time of one fold, an estimate of the task's time left (the current unit's remaining
folds at that pace, later units at the job's mean), the task's time against its limit
(`NEAR LIMIT` past 85%, `MAY HIT LIMIT` when the estimate passes it) and its memory (`sstat`
MaxRSS). The fold columns exist only for per-fold units; a fixed-protocol unit logs no per-fold
line and shows `-`.

The pace is measured, not divided out: the loop notes the log's modification time each time a
unit's fold count grows (thesis writes nothing between two fold lines, so that time is when the
fold finished), and the pace is the time between two such readings. Dividing the unit's time by
its folds would count the unit's one-off costs (loading, the WORLD resynthesis) as fold time: on
2026-11-23 a WORLD fusion unit had 1 fold after 1 h 56 min, and that division projected 1.2 d for
a unit of about 11 h, with a false `MAY HIT LIMIT`. So `per fold` and `~task left` show `-`
until the loop has seen a unit finish two folds; `--once` never has. Do not
wrap it in a loop of your own: repeated logins trip the cluster's Fail2Ban.

**A trial run.** `THESIS_UNITS=<unit>,<unit>` restricts a phase to the units it names, and the
array must then be `0..(count − 1)`. These are real units with real markers, so a later
whole-phase run skips them. Example:
`THESIS_PHASE=perfold THESIS_UNITS=pf01_dsnn_voice_dep.rep0 sbatch --array=0-0 …`.

**Measured costs.**

| | Value | Source |
|---|---|---|
| Warm array | 11 tasks, 8 min (EEG views) to 3 h 47 min (voice views), 552 cached matrices | job 5331039, 2026-11-19 |
| `perfold` unit (one repetition), with the cache | mean 2.4 h, median 3.0 h, max 3.7 h; 300 units, ~710 CPU·h | job 5331068, 2026-11-19 |
| `pfbiomex` unit (one repetition) | mean 7.2 h, median 9.9 h, max 11.2 h (the degraded-voice units are the slowest); 144 units, ~1 036 CPU·h; peak 9.3 GB | jobs 5335100 + 5338456, 2026-11-23 |
| `pfworld` unit (a whole profile, 3 repetitions) | mean 10.8 h, max 12.1 h; 150 units, ~1 614 CPU·h; peak 6.0 GB | job 5345042, 2026-11-24 |
| `pfbxworld` unit (one repetition) | mean 11.2 h, max 12.5 h; peak 9.3 GB | job 5345820, first 208 of 270 units, 2026-11-25 |
| Same voice unit, without the cache | 14 300–25 000 s, of which ~12 350 s is the handcrafted extraction | debug run, 2026-11-18 |

**What can stop a phase** (each is a loud stop, never a skip):

| Situation | Behaviour |
|---|---|
| A unit needs a cached matrix that is missing | The unit stops with `no cached features for …`. Run the warm array for **this** deploy. |
| The phase directory holds markers of another deploy (for example the 2026-11-18 debug units of `74b80587`) | The task is refused. Move the directory aside on the cluster first. |
| A redeploy | The cache key includes the revision, so the old matrices no longer match. Run the warm array again. |

**Reading the results.** Each unit writes the usual files plus
`e05_<tag>_selection_extractors.csv` and `e05_<tag>_selection_grid.csv`, and its summary says
`"protocol": "per-fold"` ([outputs table](#per-run-diagnostics-output-files)). When the
`perfold` array is home:

```bash
python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py --protocol per-fold \
  --profiles-dir src/experiments/thesis/profiles/perfold --results-dir results/thesis/perfold \
  --fixed-profiles-dir src/experiments/thesis/profiles/phase01 \
  --fixed-results-dir results/thesis/phase01 \
  --tables-dir ../../documentation/00-thesis/monography/tables
scripts/pipeline/thesis/10_thesis_perfold_significance.sh \
  src/experiments/thesis/profiles/perfold results/thesis/perfold
```

The first writes the same five tables as step 6, prefixed `perfold_` (the grid table has
one cell per source and text mode: the selection replaces nested CV, and raw features stay a
fixed-grid ablation), plus two more:

| Table | Content |
|---|---|
| `perfold_selection.csv` | per cell, what its 15 folds (3 repetitions × 5) chose: the most chosen extractor and how often (`Haar + Linear (c1) (6/15)`), how many distinct extractors, the same for the grid point (`width/blocks/lr/decay`); a tie of more than two is counted (`5 empatados (2/15 cada)`), never broken |
| `perfold_vs_fixed.csv` | every DSNN/ANN/cosine cell's per-fold EER beside the fixed grid's flat standardized cell, and the difference; descriptive (no test is declared for it) |

In `perfold_extractors.csv` the extractor columns are the folds' most frequent choice, since
each fold chooses its own. The script refuses a summary whose `protocol` disagrees with the
profile. `10_thesis_perfold_significance.sh` runs `03_`'s families unchanged on the `pf01_*`
names (§5 of the pre-registration: same comparisons, same statistics). A per-fold profile pairs
with a flat fixed profile, because its outer folds are the flat CV's.

The BIOMEX-DB replication has the same two protocols. With `pfbiomex` home:

```bash
python3 scripts/pipeline/thesis/thesis_build_biomex_tables.py --protocol per-fold \
  --profiles-dir src/experiments/thesis/profiles/pfbiomex --results-dir results/thesis/pfbiomex \
  --coretto-profiles-dir src/experiments/thesis/profiles/perfold \
  --coretto-results-dir results/thesis/perfold \
  --tables-dir ../../documentation/00-thesis/monography/tables
scripts/pipeline/thesis/11_thesis_pfbiomex_significance.sh \
  src/experiments/thesis/profiles/pfbiomex results/thesis/pfbiomex
```

It writes `pfbiomex_{auth,dysphonia,selection}.csv`. Each BIOMEX-DB cell sits beside the Coretto
**per-fold** cell, not the fixed one: the replication repeats the method, so the base it is
compared with is read under the same method. `11_` is `04_`'s families on the `pfbx01_*` names.
The memory note for the cluster: a pfbiomex task needs more than 8 GB (the degraded-voice
fusion units peaked at 9.3 GB); the sbatch now asks for 16G, but a deploy older than
2026-11-23 still has 8G, so pass `--mem=16G` on the `sbatch` line rather than redeploying
(a redeploy changes the revision, which invalidates the feature cache and the phase markers).

**Locally** (a smoke check, or a debugging run) the binary runs a per-fold profile without the
cache when `THESIS_FEATURE_CACHE` is unset. It then extracts the candidates itself, which takes
hours for a voice profile. The smoke mirror (`profiles/smoke/`, 240 samples, 2 folds, one grid
point) takes minutes:

```bash
$BIN/thesis --config src/experiments/thesis/profiles/smoke/perfold/pf01_cosine_eegpron_dep.json --repeat 0
```

## Feature cache (per-fold phases only)

**The problem.** Every per-fold unit ranks all Phase 00 candidates of its signal. For the voice
that means 153 candidates, 138 of them handcrafted, at about 3.4 h of one CPU per unit. Those
138 matrices depend on neither the fold nor the seed: they are the same in every unit that loads
the same samples. Without a cache, `perfold` alone would repeat them 300 times.

**What it does.** `thesis --warm-feature-cache` (driven by
`thesis_warm_feature_cache.sbatch`) computes each matrix once per *data view* and writes it to
`results/thesis/feature_cache/<hash>.e05fc`. There are 11 views: Coretto voice, eegpron,
eegimag and the 4 fusions, and BIOMEX-DB voice, eegpron and the 2 pronounced fusions. The phases
export `THESIS_FEATURE_CACHE_MODE=read` and only read.

```
warm array (write)                         perfold / pfbiomex / … units (read)
view "Coretto voice" ─► 138 matrices ─┐    unit pf01_dsnn_voice_dep.rep0 ─► 138 reads
view "Coretto eegpron" ─► 61 matrices ─┼─► results/thesis/feature_cache/   (a miss stops the unit)
…  (11 views)                         ─┘
```

**Why it cannot serve wrong features.** The key of a matrix is made of four parts:

| Key part | Prevents |
|---|---|
| deploy revision (`THESIS_FEATURE_CACHE_REVISION`) | features of another code version |
| fingerprint of every loaded audio/EEG value, label and sample rate | features of another base, condition or sample set |
| signal (voice / EEG) | one signal's matrix read as the other's |
| the candidate's whole `feature_extraction` block | one extractor's matrix read as another's |

The file stores its full key and a checksum. A different key, a truncated file or a wrong
checksum is refused with the remedy (delete the file and warm again). Writes go to a temp file
named `.tmp.<host>.<pid>` and are then renamed, so a reader sees a whole file or none. Two
writers of the same matrix race harmlessly. Only handcrafted candidates are cached: PCA and the
autoencoders are fitted per fold. Verified: a per-fold run gives bit-identical choices and EERs
with and without the cache (`FeatureDiskCache.PerFoldRunIsTheSameWithAndWithoutIt`, plus the
real Coretto EEG smoke run: EER 0.310185 both ways).

**Variables** (the sbatch scripts set them; set them yourself only for a local run with a cache):

| Variable | Value |
|---|---|
| `THESIS_FEATURE_CACHE` | the cache directory; unset = no cache |
| `THESIS_FEATURE_CACHE_REVISION` | required when the directory is set; the deploy's revision label |
| `THESIS_FEATURE_CACHE_MODE` | `write` (warm) or `read` (phases); anything else is refused |

A fixed-configuration profile refuses the cache, because it never ranks candidates.

## Decoded-dataset cache

The first profile of a run decodes every subject's audio+EEG from source
(`load_dataset` → `.mat`/sqlite), which dominates startup and is identical for
every profile and repeat since the raw set never changes. `ThesisDataset.cpp`
caches the fully-decoded sample set to a flat binary sidecar next to the
dataset root, one per EEG condition (`<root>.pronounced.e05dscache`,
`<root>.imagined-<seed>.e05dscache`), on the first run and reloads it directly
afterwards — no re-decode. Measured on the 1974-sample set: **~27 s cold →
~1.5 s warm** (the load phase itself drops from ~25 s to sub-second). The OS
page cache keeps the file resident across the runner's parallel workers.

- The cache stores the **full** set (before `max_samples`), so one file serves
  every `max_samples` value; truncation is always applied after loading.
- **Invalidation** is automatic: a structural signature (each source file's
  size + mtime, computed from the cheap subject-discovery pass) is stored in
  the header and re-checked on load. Editing/replacing the dataset re-decodes
  and rewrites. With a single-`.sqlite` root, the signature keys on that file.
- Writes are atomic (per-process temp + rename), so parallel cold-starts and
  crashes never leave a torn cache; any mismatch or read error silently falls
  back to a normal decode. The file is ~1.6 GB for the full set — it lives
  beside the dataset, not in the repo.
- Set **`THESIS_NO_DATASET_CACHE=1`** to bypass entirely.

## Per-run diagnostics (output files)

Every run writes into its profile's `results_dir` (`results/thesis/{phase00,phase01}`),
prefixed `e05_<run_tag>`. Alongside the metrics, each run now records the same
class of run diagnostics the Meeting01/Meeting01 pipeline stores:

| File | Contents |
|---|---|
| `*_summary.json` | Self-describing run record. Config echo (seed, modality, strategy, resolved training block) **plus** `config_hash` (provenance/determinism fingerprint), and per feature set: `param_count`, `mean_train_ms`, `mean_infer_ms`; per fold: `train_ms`, `infer_ms`, `epochs_run`, `final_train_loss`/`final_val_loss`. |
| `*_metrics.csv` | Per-fold metrics + `train_ms,infer_ms` columns. |
| `*_learning_curves.dat` | The Meeting01 epoch-history analog: one row per `(feature_set, fold, epoch)` — `train_loss val_loss epoch_ms spike_rate sops`. For the **DSNN** classifier, `spike_rate` (mean firing rate over all spiking layers) and `sops` (synaptic ops/sample) are populated per epoch; for the non-spiking RNN they are `nan`/`0`. pgfplots-ready. Written only when a classifier trained (phase01). |
| `*_paraconsistent.csv`, `*_comparison.dat` | Ranking scores and the pgfplots comparison table (unchanged). |
| `*_selection_extractors.csv` | Per-fold protocol only: `feature_set,fold,candidate,label,d_penalized,chosen`, one row per (fold, candidate). |
| `*_selection_grid.csv` | Per-fold protocol only: one row per (fold, grid point, inner fold) with the point's hyperparameters, `param_count`, `inner_eer`, `mean_inner_eer`, `chosen`. Header only for the cosine control, which has no grid. |

A per-fold summary says `"protocol": "per-fold"` (`"fixed"` everywhere else). It has no
`strategy`, and its run-level `param_count` is absent: each fold's winner has its own, under
`fold_models[f].selection`.

> **SNN firing rate is measured via a model-side probe.** The DSNN classifier trains
> under cross-entropy (not a spike loss), so the Trainer can't read the rate from the
> loss. Instead `ThesisDsnnClassifier` exposes `mean_spike_rate()`/`sops()` (computed from
> the spike trains it already caches for firing-rate regularization), and the Trainer
> populates `EpochResult` from the model when the loss doesn't. The summary's
> `mean_spike_rate`/`final_sops` are the last epoch's values, mean-aggregated over folds.

**Cross-profile significance** (the Meeting01 SNN-vs-LSTM analog). An Thesis run scores one feature
set, so the significance comparison is *across* profiles, not within a run:
`scripts/pipeline/thesis/thesis_cross_profile_significance.py` collates every profile's per-fold
`*_metrics.csv` (exactly the files each profile declares), ranks them by a metric, and tests
each condition against a declared reference — Cohen's d plus a paired Wilcoxon or an
independent Mann-Whitney U. Needs `python3` (system scipy). Example:

```bash
python3 scripts/pipeline/thesis/thesis_cross_profile_significance.py \
  --profiles-dir src/experiments/thesis/profiles/phase01 \
  --results-dir  results/thesis/phase01 \
  --metric eer --reference best --pairing independent \
  --out results/thesis/phase01/cross_profile_significance.csv
```

Every choice is required: `--metric eer|auc|accuracy|f1`, `--reference best|<run_tag>`, `--pairing paired|independent`. `paired` (Wilcoxon on (repetition, fold)-matched values) is refused for any pair that does not hold out the same folds (same `dataset.root`, `max_samples`, `seed`, `repeats`, `seed_deterministic`, `k_folds`, `nested_cv`, `text_mode`) — so across the whole Phase 01 grid, whose text modes differ, use `independent` (Mann-Whitney U). A missing metrics file, a NaN fold (closed-set metrics under verification: use eer/auc) or an unparseable value is refused, not dropped; an undefined statistic (tied values, zero variance) is an empty cell with a `note`, never p = 1 or d = 0. The comparisons share the reference, so they are one family: `p_holm` is Holm-Bonferroni over the tested comparisons and `significant_0.05` reads it. Holm does not undo selection — `--reference best` picks the reference from the same data; name a reference fixed in advance for a confirmatory reading.

## Notes

- **`run_thesis_profiles.sh all`** runs phase00 + phase01 + `debug.json` in one pass (not
  biomex). Phase 01 in that pass trains on the *previous* winners, because the post-processing
  writes the new ones only after the pass. It is a plumbing check; for results use
  `start-over-all` or the phases one by one.
- **Cost.** Every profile runs `experiment.repeats` times (3 in the profiles of record).
  The expensive-experiment guard applies: run the real phases yourself, in a terminal or in
  the background. The cluster runs and their measured durations are in
  [Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12).
- **Metrics** (verification protocol): EER and AUC are primary; closed-set metrics are `NaN`
  (correct — test speakers are unseen). MSE is autoencoder reconstruction only.

## Related

- [Experiment05](../Experiments/Thesis.md)
- [Re-run Runbook](./Re-run-Runbook.md) — the full re-run sequence this runner is one step of
- [Engineering Fixes Log](./Engineering-Fixes-Log.md) — why the automatic post-processing and the `d_penalized` ranking exist
- [Ground-Truth and Smoke Testing](./Ground-Truth-and-Smoke-Testing.md)
- [Memory Diagnostics](./Memory-Diagnostics.md)
