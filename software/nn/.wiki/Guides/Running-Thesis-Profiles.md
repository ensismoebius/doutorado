# Running Thesis Profiles (Experiment05)

How to run the thesis pipeline end-to-end: Phase 00 (feature-vector construction +
paraconsistent ranking) → Phase 01 (DSNN authentication) → the BIOMEX-DB replication. All
commands run from `software/nn/`. What happens to the results afterwards (tables, tests,
numbers in the thesis text) is in [Thesis Document](./Thesis-Document.md). See [Experiment05](../Experiments/Thesis.md)
for the design.

## Prerequisites

- The bases at the paths the profiles name: `~/database.sqlite` (Coretto; Phase 00/01) and
  `~/biomex.sqlite` (BIOMEX-DB). How to build both: [Data Bases](./Data-Bases.md).
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
then: the thesis numbers (Thesis Document guide)
```

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
  --winners results/thesis/phase00/winners.json --fused voice
```

`--fused voice|eeg` (required) chooses which signal's winner backs the fused profiles; the run of
record uses `voice`. Single-signal profiles get their own signal's winner, and
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
python3 scripts/pipeline/thesis/thesis_build_phase01_auth_tables.py \
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
winners — nothing is selected on BIOMEX-DB). Needs `~/biomex.sqlite`.

### 8. BIOMEX tables and significance

```bash
python3 scripts/pipeline/thesis/thesis_build_biomex_tables.py \
  --profiles-dir src/experiments/thesis/profiles/biomex --results-dir results/thesis/biomex \
  --coretto-profiles-dir src/experiments/thesis/profiles/phase01 \
  --coretto-results-dir results/thesis/phase01 \
  --tables-dir ../../documentation/00-thesis/monography/tables
scripts/pipeline/thesis/04_thesis_biomex_significance.sh \
  src/experiments/thesis/profiles/biomex results/thesis/biomex
```

Writes `tables/biomex_{auth,dysphonia}.csv` (each BIOMEX cell beside its Coretto cell) and
`results/thesis/biomex/significance_*.csv` (the Coretto families, declared before reading).

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
