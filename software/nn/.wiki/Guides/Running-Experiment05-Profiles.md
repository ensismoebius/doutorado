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
`[i/N] pass=… fail=… mm:ss eta~mm:ss running: <profile>`. On a terminal the
binary's own live progress bars then render underneath — **dataset loading,
feature extraction, training epochs, and CV folds** (driven by `ProgressManager`
/ `ProgressCallback`) — while output is still captured for failure diagnosis. In
a pipe/CI the ANSI bars are redirected to keep logs clean. Failures print
immediately and are listed in the final summary.

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
`results/run_profiles_<scope>.state`. Because the binary writes each profile's
result files (CSV/JSON) to disk as it finishes, a power loss only costs the
profile that was mid-run — it isn't recorded and re-runs on restart.

On start, if a state file exists the runner reports how many profiles are already
done and asks:

```
Found a previous run: 147 profile(s) already completed → results/run_profiles_phase00.state
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

Runs the 282 phase00 profiles (`classifier.enabled=false`; each extracts features
and scores them, no classifier). Writes `results/phase00/*_paraconsistent.csv`.

```bash
# heavy — background it
nohup ./scripts/testing/run_e05_profiles.sh phase00 > phase00_run.log 2>&1 &
```

### 2. Rank → winner per signal

```bash
.venv/bin/python scripts/pipeline/e05/01_e05_phase00_rank.py \
  --results-dir results/phase00 --out results/phase00/winners.json
```

Prints the ranked table and the lowest-`D_truth` winner for `voice` and `eeg`,
and writes `winners.json` (carries each winner's full `feature_extraction` block).

### 3. Inject the winner into the Phase 01 profiles

The shipped phase01 profiles carry a **placeholder** extractor until this runs.

```bash
.venv/bin/python scripts/pipeline/e05/02_e05_apply_winner.py \
  --winners results/phase00/winners.json --fused voice   # or: --fused eeg
```

`voice→voice winner`, `eeg→eeg winner`, `fused-*→--fused` (default voice, the
audio-dominant signal). Idempotent; supports `--dry-run`.

### 4. Phase 01 — DSNN authentication

Runs the 32 phase01 profiles (`classifier.enabled=true`, `paraconsistent.enabled=false`).
Writes EER/AUC to `results/phase01/`.

```bash
./scripts/testing/run_e05_profiles.sh phase01
```

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

## Notes

- **`run_e05_profiles.sh all`** runs phase00 + phase01 + debug but **skips steps 2–3**,
  so phase01 uses its placeholder extractor. For real results, run the phases
  separately with rank + apply in between.
- **Cost**: phase00 is 282 profiles × `experiment.repeats`; this is the
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
