# GridUnesp Deployment

**The problem this solves.** The `meeting01` nested-LOSO run (see
[Meeting01](../Experiments/Meeting01.md)) independently NSGA-II-searches 4 trained
families (SNN-AE, LSTM-AE, GRU-AE, Transformer-AE) across 4 datasets × 6 folds × 5
repeat runs on one 12-core desktop — worst-case ~20,635 CPU-hours (~860 days /
~28.7 months; see `meeting01-loso.json`'s `_total_runs_breakdown` for the method,
cited here rather than recomputed). UNESP's own HPC cluster,
**GridUnesp** (NCC/UNESP, Barra Funda campus), is free for UNESP researchers/students
and gives 28-core nodes with a 30-day queue, which is a real speedup for this
CPU-bound, embarrassingly-parallel-across-`(dataset,fold)` workload. This page is the
runbook for getting the existing `01_meeting01_run_loso.sh` running there unmodified
in its actual training loop, plus what had to change and what is still unconfirmed.

> Facts below marked **confirmed** came from GridUnesp's own docs. Two fetches:
> `ncc.unesp.br/gridunesp/docs/v2/` (2026-09-16) and a recheck against
> `ncc.unesp.br/gridunesp/docs/v3/pt_BR/manual_do_usuario.html` (2026-09-23) that
> caught and corrected several v2-derived numbers below (login hostname, `/home`
> and `/store` sizes) — see the per-fact notes. The v3 recheck was done through a
> page-summarizing fetch tool on pages the tool itself reported as too large to
> return verbatim in full (127k–192k characters each); specific numbers and quotes
> below sourced from v3 are therefore a close paraphrase relayed through that
> summarizer, not independently character-verified against the HTML — solid enough
> to act on, but re-verify with `support.ncc@unesp.br` if a number here and the
> cluster's actual behavior ever disagree. Facts marked **unconfirmed** were not
> answered by either version of the docs — verify with `support.ncc@unesp.br`
> before the first long submission; every step below is designed to fail safe
> (interruptible, `RESUME=1`-restartable) if a guess is wrong.

## 0. Get an account (human step, do this first)

1. Project registration: the **project coordinator must be UNESP-affiliated**
   (typically your orientador) and registers the project at
   `unesp.br/portal#!/gridunesp/submissao-de-projetos/`, or by emailing
   `grid@ncc.unesp.br`.
2. Once the project is approved, register yourself as a user at
   `ncc.unesp.br/registration/`; the coordinator gets an email to confirm you belong
   to the project before your account is finalized.
3. v2 had no stated turnaround time; the v3 manual (2026-09-23 recheck) gives a
   typical turnaround of **1–2 business days** for the approval flow (coordinator
   confirms your data by email, then the GridUnesp team finalizes the account) — but
   that number came through an AI-summarized page fetch, not verbatim text, so
   still start this well before you need the cluster rather than planning tightly
   around it.
4. v3 also confirms external collaborators with no direct UNESP affiliation can
   register as users, as long as they're linked to a registered project (only the
   project **coordinator** must be UNESP-affiliated, per point 1).

## 1. Cluster facts (confirmed)

| | |
|---|---|
| Scheduler | Slurm (`sbatch`, `#SBATCH` directives) |
| Compute nodes | 56 nodes, 2× Intel Xeon E5-2680 v4 @2.4GHz each (28 cores/node), 128GB RAM/node (4GB/core), 40GbE. **v3-new finding (2026-09-23)**: only ~26 cores/node are actually schedulable — 2 are reserved for the OS. `--cpus-per-task=28` below (build step and the sbatch job) may not be satisfiable; if either step hangs in `PENDING` instead of starting, try `--cpus-per-task=26` first. |
| GPU | one node, 4× NVIDIA L40S 48GB, `gpu` partition, 24h cap — **not needed**: `meeting01` uses the CPU/xtensor backend only |
| Queues | `short` (24h, default), `medium` (7d), `long` (30d, cluster-wide hard max), `gpu` (24h) |
| Storage | `/home` 120TB shared (v2 said 40TB — wrong, corrected by v3), `/store` 7TB shared (v2 said 32TB — wrong, corrected by v3; job-nanny's `LARGE_FILES=true` stages here too, alongside `SHARED_FS=true`), `/tmp` ~120–180GB/node local scratch (v3's own pages disagree with each other on this number — check `df -h /tmp` on a compute node rather than trusting either). **No backup, no quota on any of the three** (v3-confirmed: 100% user responsibility for data). |
| Login node | `access.grid.unesp.br` (v2 said `access2.grid.unesp.br` — v3 does not mention an "access2"; use `access.grid.unesp.br`, the scp target too) — explicitly **not** for running jobs, only preparing/submitting them. Repeated rapid `scp`/login attempts trigger a 15-minute Fail2Ban lockout, and reconnecting *during* the lockout **restarts the 15-minute timer** — if locked out, wait it out rather than retrying (v3-confirmed; use one `rsync -avz` or an archived transfer instead of many small `scp` calls). |
| Modules | Environment Modules on AlmaLinux, implemented via Lmod (v3-confirmed); `module avail` / `load` / `list` / `purge` / `show` |
| File transfer | `scp <file> user@access.grid.unesp.br:/home/user/.` or `rsync -avz` (both documented; hostname corrected 2026-09-23, see Login node row above) |

## 2. Toolchain gap — why `module load gcc` alone is not enough

The top-level `CMakeLists.txt` → `cmake/PackageChecking.cmake` hard-requires
(`REQUIRED`, unconditional, for **every** target including `meeting01` — see
[Build System](./Build-System.md)):

- `find_package(BLAS REQUIRED)` / `find_package(LAPACK REQUIRED)` /
  `pkg_check_modules(OPENBLAS REQUIRED openblas)`

(An `SDL2 REQUIRED` check used to be here too, hard-required project-wide despite
having zero actual consumers anywhere in the codebase — no target linked it.
Removed at the source 2026-09-16 rather than worked around, so this is one less
thing to install everywhere, not just on GridUnesp. The project's only GUI demo,
`snn_spike_plotter` — GLFW+OpenGL3+Dear ImGui — was itself removed later the same
week as unused, so the project now has no GUI backend at all and no windowing
dependency of any kind.)

GridUnesp's module list has **no OpenBLAS-via-pkg-config** module (only Intel MKL,
which is a different discovery path) — this alone still forces the conda env below
regardless of the `cmake`/`ninja` points that follow. It also has **no `ninja`
module** (the `max-performance` preset's generator; v3-confirmed still absent,
2026-09-23 recheck) — this was `cmake/3.9.0` (default) < `cmake_minimum_required
(VERSION 3.10)`, `cmake/3.20.0-rc3` the only one clearing it, as of the 2026-09-16
v2 check. **v3 (2026-09-23) lists a newer `cmake/4.0.3` module** not present in v2 —
worth using instead of `cmake/3.20.0-rc3` if a future setup drops the conda env's
own `cmake` package, though the OpenBLAS gap alone means conda is still required
today either way, so this is not acted on here.

Fix: `scripts/pipeline/meeting01/gridunesp_setup_env.sh` — a conda env
(`meeting01-build`) with `openblas pkg-config ninja git cmake ccache zlib hdf5
fftw sqlite make` and a pinned GCC 13 (`gxx_linux-64`/`gcc_linux-64`), so nothing
in `PackageChecking.cmake` needs a cluster-specific carve-out and the build stays
identical to the local one. `zlib` is defensive — `find_package(ZLIB REQUIRED)`
in `src/core/data_loaders/CMakeLists.txt` has no vendored fallback — most Linux
base images already have it, but it costs nothing to guarantee. `hdf5` is
**not** defensive — vendored `matio` (`cmake/VendorMatio.cmake`) hard-requires it
for MAT73 support with no fallback, and its absence is a fatal configure error,
not a warning. `fftw` avoids an unnecessary from-source vendored FFTW3 build
(`cmake/VendorFFTW.cmake` falls back to it automatically, just slower). `sqlite`
matters for a subtler reason: `cmake/VendorSqlite.cmake` prefers
`find_package(SQLite3 QUIET)` and only falls back to downloading a vendored
amalgamation from hardcoded, **non-year-prefixed** sqlite.org URLs if that fails
— and those URLs 404 the moment sqlite.org ships a newer point release and moves
the old one into a dated subdirectory. Without a system SQLite3, that fallback is
a live 404, not a safety net. `make` is required because neither
`gcc_linux-64`/`gxx_linux-64` nor a minimal AlmaLinux base ship GNU make, and two
things silently need it: the vendored NFFT3 autotools build
(`cmake/VendorNFFT3.cmake`), and — much less obviously — GCC's own `-flto=auto`
(part of the `max-performance` preset), which spawns parallel LTRANS jobs via an
internal `make -jN`; without it the failure surfaces deep inside
collect2/lto-wrapper as `lto-wrapper: fatal error: execvp: No such file or
directory`, a message that names neither "make" nor anything else recognisable.

The GCC version matters more than it looks. GCC 10 satisfies the project's
`requires(...)` concepts floor (`Tensor.hpp`, `Linear.hpp`, `Lif.hpp`,
`Adam.hpp` need real C++20 concepts, not the older Concepts TS) — but concepts
support is not the project's actual minimum. `Meeting01Config.cpp` uses
`std::ostringstream::view()`, a separate C++20 **library** feature (P2495) that
GCC 10's libstdc++ does not implement, one compiler version short of where
concepts support lands. Building with `gxx_linux-64=10` compiles 90 of 138 build
steps — including code that uses concepts — before failing with `error:
'std::ostringstream' has no member named 'view'`, which is easy to misdiagnose
as a concepts problem since everything upstream of it that *does* use concepts
compiles fine. Verified empirically (not assumed) that GCC 13's libstdc++ has
`ostringstream::view()` and GCC 10's does not.

All five gaps (`hdf5`, `fftw`, `sqlite`, `make`, and the GCC 10→13 bump) were
caught by `scripts/pipeline/meeting01/run_gridunesp_docker_sim.sh` (see below)
before ever touching the real cluster.

```bash
module load miniconda/24.4.0-libmamba
./scripts/pipeline/meeting01/gridunesp_setup_env.sh
```

Every later shell (configure, build, and inside the sbatch job) needs:

```bash
module load miniconda/24.4.0-libmamba
conda activate meeting01-build
```

### Validating this locally before submitting

`scripts/pipeline/meeting01/run_gridunesp_docker_sim.sh` reproduces the toolchain
gap above in a local AlmaLinux 9 + conda container (`Dockerfile.gridunesp-sim`) and
runs `gridunesp_setup_env.sh` **unmodified**, then `cmake --preset=max-performance`
and `cmake --build ... --target meeting01`, ending with a `meeting01 --help` smoke
check — all without touching the host's own `out/` build tree or spending any
GridUnesp queue time. It caught both the `hdf5`/`fftw` gap above and a
`CondaToSNonInteractiveError` (recent conda refuses to run non-interactively unless
the `defaults` channels' Terms of Service are accepted, even though this script
only ever installs from `conda-forge` — fixed with `--override-channels`) before
either one ever reached the real cluster.

```bash
./scripts/pipeline/meeting01/run_gridunesp_docker_sim.sh
```

It does **not** simulate Slurm/`sbatch`/`job-nanny`, GridUnesp's actual Environment
Modules, or `-march=native` codegen for the cluster's specific Xeon E5-2680 v4 —
see the Dockerfile's own header comment for the full boundary of what this checks.

## 3. One-time setup: datasets, configure, build

**Fast path**: `scripts/pipeline/meeting01/gridunesp_deploy.sh` runs everything in
this whole section — syncs the checkout to GridUnesp, bootstraps the conda env,
fetches all 4 datasets, configures, and builds on a compute node — as one command
from the local machine. Safe to re-run any time (every step it wraps is itself
idempotent). It deliberately stops short of submitting the real job: it prints the
exact `sbatch` command from §4 below and exits, so starting the actual multi-week
run is still a decision you make explicitly.

```bash
GRIDUNESP_USER=<your grid username> ./scripts/pipeline/meeting01/gridunesp_deploy.sh
```

The rest of this section explains what it does and why, step by step — read on if
it fails partway and you need to debug a specific stage, or if you'd rather run the
steps by hand.

**Datasets first** (~24GB total: 27MB FSDD + 357MB AudioMNIST (resampled from a 9.4GB
48kHz clone) + 3.4GB eegmmidb + 20.3GB Siena Scalp EEG, well under `/home`'s 120TB
and — per the v3 manual — not quota-limited at all). The profile
(`meeting01-loso.json`) hardcodes absolute dataset paths for
reproducibility, so this only reproduces the expected layout correctly if the
GridUnesp account's username is also `ensismoebius` (same caveat the old manual-`scp`
approach below had) — otherwise fork the profile's `dataset.sources[].root` paths to
match the real account home first.

`scripts/pipeline/meeting01/ensure_datasets.sh` fetches all four datasets directly on
whichever machine it runs on — idempotent (safe to re-run; skips anything already
complete) and fail-loud (a bad clone or an interrupted transfer is a hard error, not a
silent partial dataset). Run it **on the login node** (confirmed internet; compute-node
internet is unconfirmed, see below) instead of the old "scp a pre-populated tree from
another machine" approach — a fresh GridUnesp checkout no longer depends on the local
machine having downloaded everything first:

```bash
module load miniconda/24.4.0-libmamba && conda activate meeting01-build   # needs git/wget/sox
./scripts/pipeline/meeting01/ensure_datasets.sh
```

(The old approach — `scp -r` a pre-populated `databases/` tree from the local machine —
still works if you already have one locally and would rather not re-download 24GB on
the cluster; either gets to the same on-disk layout.)

**Configure on the login node** (needs internet — every tensor/matio/cnpy dependency
is `FetchContent`-cloned from GitHub during configure; **unconfirmed** whether compute
nodes have outbound internet at all, so do this step, which is preparation rather
than "running a simulation," on the login node where internet is known to work):

```bash
cd software/nn
module load miniconda/24.4.0-libmamba && conda activate meeting01-build
cmake --preset=max-performance
```

**Build on a compute node**, not the login node — `max-performance` compiles with
`-march=native`, so the binary must be *compiled* on the same CPU model it will
*run* on (Xeon E5-2680 v4), not whatever the login node happens to be:

```bash
srun --partition=short --time=00:30:00 --cpus-per-task=28 --pty bash
module load miniconda/24.4.0-libmamba && conda activate meeting01-build
cmake --build out/build/max-performance --target meeting01 -j28
exit
```

By this point dependencies are already fetched (step above), so this build needs no
internet — safe regardless of the unconfirmed compute-node connectivity question.

Sanity-check before submitting the long job:

```bash
srun --partition=short --time=00:10:00 --cpus-per-task=4 --pty \
  out/build/max-performance/src/experiments/meeting01/meeting01 --help
```

## 4. Submitting the run

```bash
sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
squeue -u $USER
```

The script (`01_meeting01_run_loso_gridunesp.sbatch`) sets `EXPERIMENT_CONFIRMED=1
SKIP_BUILD=1 SKIP_POSTPROCESS=1` and calls the **unmodified**
`01_meeting01_run_loso.sh` under `job-nanny` (GridUnesp's required job wrapper).
`SKIP_POSTPROCESS=1` (new flag, this session) stops the script after the training
loop — the paper-table scripts (`03_`/`02_`/`04_`) default `--data-dir` to an absolute
path under the *local* machine's `documentation/` tree, which does not exist on
GridUnesp and does not need to (see §5).

To continue an interrupted or partially-complete run:

```bash
RESUME=1 sbatch scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
```

`RESUME=1` uses the training loop's existing semantics — skips straight past any
`(dataset, fold)` whose `*_comparative_metrics.csv` already exists, unchanged from
local-machine behavior (see [Meeting01](../Experiments/Meeting01.md#running-it)).

**Monitoring progress remotely.** `monitor.py` (the same live dashboard used for a
local run — see [Meeting01 § Running it](../Experiments/Meeting01.md#running-it))
already works unmodified against a GridUnesp run; two thin wrappers make it easy to
reach from the local machine without an interactive login shell each time:

```bash
# Live view, no local copy of the data -- one SSH session, --plain mode (stdlib
# only, no `rich`/conda env needed remotely). Runs monitor.py's own refresh loop
# INSIDE that one session rather than reconnecting repeatedly.
GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh
GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh --once   # single snapshot
GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/remote_monitor.sh --rank 3

# Or: sync results/meeting01/ down and use the local rich dashboard / archive it
GRIDUNESP_USER=<user> ./scripts/pipeline/meeting01/pull_progress.sh
.venv/bin/python3 scripts/pipeline/meeting01/monitor.py --run-tag meeting01_loso
```

Both default `GRIDUNESP_HOST=access.grid.unesp.br` and
`GRIDUNESP_REMOTE_DIR=software/nn` (override via those env vars if your remote
layout differs). Both are designed around the Fail2Ban lockout above: one SSH/rsync
connection per invocation, not a retry loop — use `monitor.py`'s own `--interval`
(inside `remote_monitor.sh`'s one session) or a real-delay `watch -n 60 ...` around
`pull_progress.sh` rather than hammering the login node.

> **Partially confirmed, one real gap still open (v3 recheck, 2026-09-23)**:
> `job-nanny`'s `SHARED_FS` flag is real and does what the sbatch script assumes —
> v3 documents it exactly (default `false`; single-node job with `SHARED_FS`/
> `LARGE_FILES` both unset stages to per-node `/tmp`; `SHARED_FS=true` — what this
> sbatch script sets — or `LARGE_FILES=true`, or a multi-node job, stages to the
> shared `/store` instead). There is also a periodic-sync mechanism, `CHECKPOINT`
> (default `$OUTPUT`) / `WAIT_CHECKPOINT` (default 10800s = 3h), which is how a
> multi-day `/tmp`-staged run's results would get copied back to `/home`
> incrementally rather than only at job end — not directly relevant here since
> `SHARED_FS=true` writes to `/store` directly, but useful context for why that flag
> matters at all.
>
> The real gap: v3 describes `INPUT` and `OUTPUT` as **required** job-nanny
> variables (files/dirs staged in before the job runs / copied back when it ends),
> and this sbatch script sets neither. Whether that is a hard failure (job-nanny
> refuses to run without them) or a soft one (only logging/staging metadata is
> incomplete, since `SHARED_FS=true` already means job-nanny isn't doing any actual
> copying) was **not resolved** by this recheck — the source page itself only
> describes `INPUT`/`OUTPUT` as "always define" without stating what happens if you
> don't. Do not trust the reading above as settled: confirm with
> `support.ncc@unesp.br`, or — cheaper — first submit a `short`-queue smoke job
> using the real sbatch script with a trivial workload (e.g. `meeting01 --help`
> instead of the full training loop) and read job-nanny's own log output for a
> complaint about missing `INPUT`/`OUTPUT` before ever submitting the real 30-day
> job. If wrong, the run is still safe to kill and `RESUME=1` restart either way.

## 5. Bringing results home

Sync back to the local checkout (merges into whatever folds already ran locally —
`RESUME=1` semantics apply on both ends, so this is safe to run mid-flight too):

```bash
rsync -avz <user>@access.grid.unesp.br:software/nn/results/meeting01/ \
  results/meeting01/
```

Then, **locally**, finish the pipeline exactly as documented in
[Meeting01 § Running it](../Experiments/Meeting01.md#running-it):

```bash
EXPERIMENT_CONFIRMED=1 RESUME=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
```

Every `(dataset, fold)` is already complete after the sync, so this call skips the
entire training loop and falls straight through to `03_`/`02_`/`04_` with correct
local absolute paths — no GridUnesp-specific path handling needed on this end.

## Forward-looking: other experiments use a different dataset mechanism

This runbook covers `meeting01` only, and `meeting01` never touches SQLite — its profile
(`meeting01-loso.json`) hardcodes absolute dataset-directory paths
(`/home/ensismoebius/.../databases/{fsdDataset,audioMNIST_8k,eegmmidb,siena}`), which is
exactly why §3 above fetches into the *same absolute path* under the grid account's
`$HOME` rather than doing anything sqlite-specific.

`thesis`, `paraconsistentGA`, and `autoencoderRunner` are a different story: they all read
a single `~/database.sqlite` file (three independent call sites —
`ThesisDataset::load_dataset`, `SqliteBatchSource`, `TrialFoldSelector` — each opens its
own `sqlite3` connection, but all resolve the path the same way, via
`nn::utility::expand_home("~/database.sqlite")`). None of these are part of the current
GridUnesp plan. If one of them is submitted to the grid later, the fix is one line, not a
code change: `scp` the local `database.sqlite` to `~/database.sqlite` on
`access.grid.unesp.br` once, and `expand_home()` picks it up automatically on every
later run, the same as any other `~`-relative path on that account.

## Open questions to confirm with `support.ncc@unesp.br`

(Rechecked against the v3 manual 2026-09-23 — items resolved by that recheck are
marked so; everything else here is still genuinely open.)

- Do **compute** nodes (not the login node) have outbound internet? **Still
  unresolved** — v3's infrastructure and FAQ pages were checked specifically for
  this and neither states it either way. Affects nothing in the plan above
  (`cmake --preset` and `ensure_datasets.sh` both run on the login node specifically
  to avoid needing this), but would simplify things if true.
- `SHARED_FS=true` itself is now v3-confirmed as the flag that routes a single-node
  job to `/store` instead of per-node `/tmp` staging — **resolved**, no longer an
  open question on its own. What's still open: whether `job-nanny`'s `INPUT`/
  `OUTPUT` variables (documented as required) can safely be left unset the way this
  sbatch script currently does — see the callout in §4 above; verify with a short
  smoke job before the first long submission.
- Realistic queue wait time for `long` (30-day) jobs — affects whether `long` or a
  shorter, resubmitted-via-`RESUME=1` `medium`/`short` chain is the better fit.
- Two internal contradictions in v3's own pages, neither resolved by this recheck:
  per-node `/tmp` capacity is stated as both ~180GB and ~120GB on different pages,
  and `/store` persistence is described as both "temporary, wiped at job end" and
  "permanent" on different pages. Neither matters for this workload today (`/tmp`
  is unused since `SHARED_FS=true`; `/store` holds only the ~24GB of datasets plus
  incrementally-written results, well under either stated capacity), but worth a
  direct `df -h` check on a compute node before relying on either number.

## Related

- [Meeting01](../Experiments/Meeting01.md) — the experiment being deployed
- [Build System](./Build-System.md) — `PackageChecking.cmake` / `Vendor*.cmake` design
- [Re-run Runbook](./Re-run-Runbook.md) — local-machine equivalent commands
