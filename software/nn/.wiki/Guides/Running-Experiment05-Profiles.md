# Running Experiment05 Profiles

How to run the Experiment05 pipeline end-to-end: Phase 00 (feature-vector
construction + paraconsistent ranking) → Phase 01 (DSNN authentication). All
commands run from `software/nn/`. See [Experiment05](../Experiments/Experiment05.md)
for the design.

## Prerequisites

- The dataset present at each profile's `dataset.root` (SQLite).
- The `experiment05` binary built in any CMake profile:

```bash
cmake --preset=max-performance
cmake --build out/build/max-performance --target experiment05 -j$(nproc)
```

## Runner scripts

Two runners share one live-progress UI, failure capture, and binary auto-detection:

| Script | Runs | Params |
|---|---|---|
| `scripts/testing/run_e05_profiles.sh [phase00\|phase01\|all]` | the **real** profiles | full (the actual experiment) |
| `scripts/testing/run_e05_smoke.sh [phase00\|phase01\|all]` | the **smoke** mirror | tiny (fast crash check) |

**Binary selection** (works with any build profile):

- auto: the most recently built `out/build/*/…/experiment05`;
- `E05_BUILD=max-performance-opencl ./scripts/testing/run_e05_profiles.sh phase00`;
- `E05_BIN=/abs/path/to/experiment05 ./scripts/testing/run_e05_profiles.sh`.

**Progress**: before each profile the runner prints
`[i/N] elapsed h:mm:ss eta~h:mm:ss running: <profile>`. On a terminal the
binary's own live progress bars then render underneath — **dataset loading,
feature extraction, training epochs, and CV folds** (driven by `ProgressManager`
/ `ProgressCallback`) — while output is still captured for failure diagnosis. In
a pipe/CI the ANSI bars are redirected to keep logs clean. Failures print
immediately and are listed in the final summary.

**Rich per-bar metadata (Guayaquil/E04 look)**: each training bar carries a
metadata line — `label │ description │ run X/Y  LOSS │ phases`. The autoencoder
feature-extraction bars name the model + loss (`SNN-AE (poisson)`, `ANN-AE
(direct)`, `LSTM-AE` │ `MSE`); the DSNN classifier bars show `run fold/total`
and `CrossEntropy` with live train/val loss. This is the same
`ProgressCallback::set_metadata` path E04 uses, so both experiments read alike.

**Overall run banner + work-weighted ETA**: each profile is a *separate process*
and cannot know the whole-run progress, so the runner computes an overall status
and passes it to the binary via `E05_OVERALL` — rendered as a persistent top
line (`Overall [i/N] elapsed … ETA …`) above the profile's own bars, mirroring
E04's `E04_OVERALL`. The overall ETA is **work-weighted + EMA-smoothed**
(`scripts/lib/run_eta.sh`): each profile is weighted by rough cost
(`p00_hc_*` = 1, autoencoders / DSNN = 20) and the runner tracks
seconds-per-unit-work rather than counting profiles equally. This matters because
the mix is wildly heterogeneous — a naive per-profile mean reads ~7× too high
through the heavy-AE phase and only corrects near the end; the weighted estimate
stays stable from the first completed AE. In the parallel monitor the same
weighting drives the dashboard's `overall ETA` (workers append their weight to
`$TMP/weights_done`; the monitor sums it each frame).

## Live dashboard (parallel runs)

With `JOBS>1` on a terminal (`E05_MONITOR` unset or `1`), the workers' own
progress bars can't share one screen, so a background `monitor_loop()`
redraws a full dashboard in place every `E05_MONITOR_INTERVAL` seconds
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
of the real terminal size. `E05_MONITOR=0` disables the dashboard in favor of
plain `start:`/`PASS`/`FAIL` event lines; `E05_MONITOR_TAIL_BYTES` /
`E05_MONITOR_HEAD_BYTES` bound how much of each (multi-MB) worker log is
re-read per frame.

Serial runs (`JOBS=1`) skip the dashboard entirely — the binary's own bars
render directly, as described above.

## Parallelism (memory-gated)

`run_e05_profiles.sh` runs several profiles at once — each is an independent
process with its own `run_tag`, so phase00/phase01 parallelise safely as far
as *correctness* goes. Job count is computed as:

```
JOBS = min(MemAvailable / E05_JOB_MEM_MB, nproc), capped at 4
```

Override with `E05_JOBS` directly, or tune the per-job budget with
`E05_JOB_MEM_MB` (default **5120**).

**2026-07-14 fix**: the default used to be 2048MB. Measured peak RSS+swap for
real phase00 profiles: `snn-ae`/poisson **voice** profiles plateau around
**4.4GB**, EEG profiles around **2.1GB** (`kAeInputFeatures=256`,
`time_steps=16`, ~1974 voice samples → 31,584 spike frames trained for
`epochs=100`). With the old 2048MB budget the script would happily launch 4
voice jobs in parallel on a machine that could only actually hold 1–2,
pushing the box into heavy swap (14GB+ swap used on a 17GB machine). The
budget is now 5120MB so `JOBS` auto-throttles correctly; if you profile a
config with a different memory footprint, pass `E05_JOB_MEM_MB` explicitly
rather than trusting the default blindly.

The 2GB-per-job assumption wasn't a leak in `experiment05` itself — see
[Core/Tensor § Recent OpenCL Buffer Pool Memory Cap](../Core/Tensor.md#recent-opencl-buffer-pool-memory-cap-2026-07-14)
for the related (secondary) buffer-pool bound that was tightened at the same
time, and [Memory Diagnostics](./Memory-Diagnostics.md) for the full
investigation writeup and the general leak-vs-plateau method.

**2026-07-15 fixes** (both in `run_e05_profiles.sh`):
- **Throttle correctness.** The parallel pool used to gate on `$TMP/active`
  marker files. A worker stopped by `SIGTTOU`/`SIGTTIN` (the script
  backgrounded while something touches the TTY) never removes its marker, and
  a signal-interrupted `wait -n` returns without reaping — so the gate spun
  through and launched **every** pending profile at once (observed twice:
  ~30 concurrent `experiment05` processes, OOM-killing unrelated work). It now
  throttles on live worker PIDs, which is immune to both.
- **Binary auto-pick.** Auto-pick now prefers the `max-performance` (CPU) build
  when present, instead of "most recently built". These profiles' networks are
  tiny and kernel-launch-bound on the GPU, so a stray OpenCL rebuild used to
  silently switch runs onto the slower backend. Override with `E05_BUILD` /
  `E05_BIN` to target a specific backend.

## Crash / power-loss recovery

`run_e05_profiles.sh` checkpoints every completed profile to
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
./scripts/testing/run_e05_smoke.sh phase00
./scripts/testing/run_e05_smoke.sh phase01
```

The smoke mirror (`profiles/smoke/`) is auto-regenerated by the CMake
`e05_smoke_profiles` target whenever a real profile changes. See
[Ground-Truth and Smoke Testing](./Ground-Truth-and-Smoke-Testing.md).

## Full pipeline

### 1. Phase 00 — feature ranking

Runs the **208** phase00 profiles (`classifier.enabled=false`; each extracts
features and scores them, no classifier — the grid was 300 before the D6 fix
retired 92 degenerate EEG bark/mel profiles, see `run.md`). Writes
`results/thesis/phase00/*_paraconsistent.csv`.

```bash
# heavy — background it
nohup ./scripts/testing/run_e05_profiles.sh phase00 > phase00_run.log 2>&1 &
```

**As of 2026-07-18, steps 2 and 3 below now run automatically** as
`post_process_phase00()`, immediately after all 208 profiles finish with zero
failures — you no longer need to invoke them by hand for a normal run. They
are documented here for manual re-runs (e.g. re-ranking after fixing a
partial failure without re-running the whole phase, or a dry-run against
`results/thesis/smoke`).

Gating on the automatic path:
- **Fails closed.** Any profile failure skips post-processing entirely — the
  ranking step selects on `d_penalized`, and computing it over an incomplete
  result set risks writing the wrong winner into all 32 tracked phase01
  profiles (step 3). Fix the failures and re-run (resume skips what already
  passed), or set `E05_FORCE_POST=1` to force it anyway.
- `E05_SKIP_POST=1` disables the automatic post-processing entirely.
- With `scope=all`, phase01 runs in the *same* pass as phase00, so it trains
  on the placeholder extractor — the winner injected by step 3 comes too late
  for that pass. The script prints this warning; run phase01 alone afterward
  (§4) for the injected winner to take effect.
- Step 3 still **rewrites the 32 phase01 profiles in place** whether it ran
  automatically or by hand — review `git diff src/experiments/05/profiles/phase01`
  before committing.

### 2. Rank → winner per signal

```bash
.venv/bin/python scripts/pipeline/e05/01_e05_phase00_rank.py \
  --profiles-dir src/experiments/05/profiles/phase00 \
  --results-dir  results/thesis/phase00 \
  --out          results/thesis/phase00/winners.json
```

Selects on `d_penalized` (not raw `d_truth`, which a collapsed latent can
game) and reports exact ties explicitly — watch for a printed `[TIE]`
warning, which means the winner was chosen by sort tie-break rather than
merit. Writes `winners.json` (carries each winner's full
`feature_extraction` block). `--profiles-dir`/`--results-dir` default to the
real phase00 paths shown above; redirect both together when ranking a
smoke/partial run — see the destructive-path warning in `run.md` §2a.

### 3. Inject the winner into the Phase 01 profiles

The shipped phase01 profiles carry a **placeholder** extractor until this runs.

```bash
.venv/bin/python scripts/pipeline/e05/02_e05_apply_winner.py \
  --winners      results/thesis/phase00/winners.json \
  --profiles-dir src/experiments/05/profiles/phase01
```

`--fused voice|eeg` (default `voice`, the audio-dominant signal) selects
which winner backs the fused profiles; `voice`/`eeg`-only profiles always get
their own signal's winner. Idempotent; supports `--dry-run`. **Rewrites the
32 tracked phase01 profiles in place** — pointing `--winners` at a
smoke/partial-run JSON silently overwrites the real profiles' extractor.

### 4. Regenerate the thesis tables

```bash
.venv/bin/python scripts/pipeline/e05/e05_build_phase00_paraconsistent_tables.py \
  --results-dir results/thesis/phase00 \
  --tables-dir  ../../documentation/00-thesis/monography/tables
```

Defaults `--tables-dir` to the committed thesis tables directory — run it
with anything other than complete phase00 results and it overwrites the
tables the thesis compiles from. Ranks **ascending** by `d_penalized` (rank 1
is the best configuration) and marks each signal's winner with a dagger —
an earlier version of this script sorted descending, which put the actual
winner last instead of first; see
[Experiment05 pitfall 12](../Experiments/Experiment05.md#common-pitfalls)
before trusting a "rank 1" row in any table this pipeline produces.

### 5. Phase 01 — DSNN authentication

Runs the 32 phase01 profiles (`classifier.enabled=true`, `paraconsistent.enabled=false`).
Writes EER/AUC to `results/thesis/phase01/`. Must run **after** step 1 (which
now runs steps 2-4 automatically on a clean pass) — otherwise it trains on
the placeholder extractor rather than the phase00 winner.

```bash
./scripts/testing/run_e05_profiles.sh phase01
```

### 6. Regenerate the thesis authentication table

Not chained automatically (unlike step 4) — run by hand once phase01 has
results:

```bash
.venv/bin/python scripts/pipeline/e05/e05_build_phase01_auth_tables.py \
  --results-dir results/thesis/phase01 \
  --tables-dir  ../../documentation/00-thesis/monography/tables
```

Averages `mean_eer`/`mean_auc` over each configuration's 3 repeats and
writes `tables/phase01_auth.csv`, sorted ascending by EER (lower is better).

> **Status (2026-07-19):** both phases have completed a full run — 208/208
> phase00 and 32/32 phase01 profiles, 0 failures. See
> [Experiment05](../Experiments/Experiment05.md#overview) for the headline
> numbers and the thesis chapter 09 for full tables and discussion.

## Decoded-dataset cache

The first profile of a run decodes every subject's audio+EEG from source
(`load_dataset` → `.mat`/sqlite), which dominates startup and is identical for
every profile and repeat since the raw set never changes. `E05Dataset.cpp`
caches the fully-decoded sample set to a flat binary sidecar next to the
dataset root (`<root>.e05dscache`) on the first run and reloads it directly
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
- Set **`E05_NO_DATASET_CACHE=1`** to bypass entirely.

## Per-run diagnostics (output files)

Every run writes into its profile's `results_dir` (`results/thesis/{phase00,phase01}`),
prefixed `e05_<run_tag>`. Alongside the metrics, each run now records the same
class of run diagnostics the Guayaquil/E04 pipeline stores:

| File | Contents |
|---|---|
| `*_summary.json` | Self-describing run record. Config echo (seed, modality, strategy, resolved training block) **plus** `config_hash` (provenance/determinism fingerprint), and per feature set: `param_count`, `mean_train_ms`, `mean_infer_ms`; per fold: `train_ms`, `infer_ms`, `epochs_run`, `final_train_loss`/`final_val_loss`. |
| `*_metrics.csv` | Per-fold metrics + `train_ms,infer_ms` columns. |
| `*_learning_curves.dat` | The E04 epoch-history analog: one row per `(feature_set, fold, epoch)` — `train_loss val_loss epoch_ms spike_rate sops`. For the **DSNN** classifier, `spike_rate` (mean firing rate over all spiking layers) and `sops` (synaptic ops/sample) are populated per epoch; for the non-spiking RNN they are `nan`/`0`. pgfplots-ready. Written only when a classifier trained (phase01). |
| `*_paraconsistent.csv`, `*_comparison.dat` | Ranking scores and the pgfplots comparison table (unchanged). |

> **SNN firing rate is measured via a model-side probe.** The DSNN classifier trains
> under cross-entropy (not a spike loss), so the Trainer can't read the rate from the
> loss. Instead `E05DsnnClassifier` exposes `mean_spike_rate()`/`sops()` (computed from
> the spike trains it already caches for firing-rate regularization), and the Trainer
> populates `EpochResult` from the model when the loss doesn't. The summary's
> `mean_spike_rate`/`final_sops` are the last epoch's values, mean-aggregated over folds.

**Cross-profile significance** (the E04 SNN-vs-LSTM analog). An E05 run scores one feature
set, so the significance comparison is *across* profiles, not within a run:
`scripts/pipeline/e05/e05_cross_profile_significance.py` collates every profile's per-fold
`*_metrics.csv`, ranks them by a metric (`--metric eer|accuracy|auc|f1`), and tests each
condition against the best — Cohen's d plus a paired Wilcoxon (same CV seed/splits) or an
independent Mann-Whitney U. Needs `python3` (system scipy). Example:
`python3 scripts/pipeline/e05/e05_cross_profile_significance.py --results-dir results/thesis/phase01 --metric eer`.

## Notes

- **`run_e05_profiles.sh all`** runs phase00 + phase01 + debug in one pass, and
  (since 2026-07-18) still auto-runs the phase00 post-processing on a clean
  pass — but phase01 in that *same* pass trains on the placeholder extractor,
  since it ran before the winner was injected. For real phase01 results after
  an `all` run, run phase01 alone again (§5). Running the phases separately
  (`phase00` then `phase01`) avoids the caveat entirely.
- **Cost**: phase00 is 208 profiles × `experiment.repeats`; this is the
  ~2.5 h-class run. The expensive-experiment guard applies — run it yourself in a
  terminal/background. Drop `experiment.repeats` to 1 if you only want a single
  pass (the rank script averages over whatever repeats exist).
- **Metrics** (verification protocol): EER and AUC are primary; closed-set metrics
  are `NaN` (correct — test speakers are unseen). MSE is autoencoder reconstruction
  only.

## Related

- [Experiment05](../Experiments/Experiment05.md)
- [Ground-Truth and Smoke Testing](./Ground-Truth-and-Smoke-Testing.md)
- [Grid Runbook](./Grid-Runbook.md)
- [Memory Diagnostics](./Memory-Diagnostics.md)
