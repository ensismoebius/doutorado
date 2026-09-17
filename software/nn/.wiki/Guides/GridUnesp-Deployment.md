# GridUnesp Deployment

**The problem this solves.** The `meeting01` nested-LOSO run (see
[Meeting01](../Experiments/Meeting01.md)) trains ~8100 configs across 3 datasets × 6
folds on one 12-core desktop — ETA measured in weeks. UNESP's own HPC cluster,
**GridUnesp** (NCC/UNESP, Barra Funda campus), is free for UNESP researchers/students
and gives 28-core nodes with a 30-day queue, which is a real speedup for this
CPU-bound, embarrassingly-parallel-across-`(dataset,fold)` workload. This page is the
runbook for getting the existing `01_meeting01_run_loso.sh` running there unmodified
in its actual training loop, plus what had to change and what is still unconfirmed.

> Facts below marked **confirmed** came from GridUnesp's own docs
> (`ncc.unesp.br/gridunesp/docs/v2/`, fetched 2026-09-16). Facts marked **unconfirmed**
> were not answered by those pages — verify with `support.ncc@unesp.br` before the
> first long submission; every step below is designed to fail safe (interruptible,
> `RESUME=1`-restartable) if a guess is wrong.

## 0. Get an account (human step, do this first)

1. Project registration: the **project coordinator must be UNESP-affiliated**
   (typically your orientador) and registers the project at
   `unesp.br/portal#!/gridunesp/submissao-de-projetos/`, or by emailing
   `grid@ncc.unesp.br`.
2. Once the project is approved, register yourself as a user at
   `ncc.unesp.br/registration/`; the coordinator gets an email to confirm you belong
   to the project before your account is finalized.
3. No stated turnaround time — start this before you need the cluster, not the week
   the local run's ETA becomes unacceptable.

## 1. Cluster facts (confirmed)

| | |
|---|---|
| Scheduler | Slurm (`sbatch`, `#SBATCH` directives) |
| Compute nodes | 56 nodes, 2× Intel Xeon E5-2680 v4 @2.4GHz each (28 cores/node), 128GB RAM/node (4GB/core), 40GbE |
| GPU | one node, 4× NVIDIA L40S 48GB, `gpu` partition, 24h cap — **not needed**: `meeting01` uses the CPU/xtensor backend only |
| Queues | `short` (24h, default), `medium` (7d), `long` (30d, cluster-wide hard max), `gpu` (24h) |
| Storage | `/home` 40TB (only mount visible from login), `/store` 32TB Lustre shared, `/tmp` 120GB/node local scratch. **No backup system.** |
| Login node | `access2.grid.unesp.br` (scp target too) — explicitly **not** for running jobs, only preparing/submitting them |
| Modules | Environment Modules on AlmaLinux; `module avail` / `load` / `list` |
| File transfer | `scp <file> user@access2.grid.unesp.br:/home/user/.` — repeated rapid `scp` calls trigger a 15-min Fail2Ban lockout |

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
which is a different discovery path). It also has **no `ninja` module** (the
`max-performance` preset's generator) and neither `cmake` module meets the project
floor (`cmake/3.9.0` default < `cmake_minimum_required(VERSION 3.10)`;
`cmake/3.20.0-rc3` is the only one that clears it).

Fix: `scripts/pipeline/meeting01/gridunesp_setup_env.sh` — a conda env
(`meeting01-build`) with `openblas pkg-config ninja git cmake ccache zlib` and a
pinned GCC 10 (`gxx_linux-64`/`gcc_linux-64`), so nothing in `PackageChecking.cmake`
needs a cluster-specific carve-out and the build stays identical to the local one.
GCC 10 was picked because the project's `requires(...)` concepts (`Tensor.hpp`,
`Linear.hpp`, `Lif.hpp`, `Adam.hpp`) need real C++20 concepts support (GCC ≥10),
not the older Concepts TS. `zlib` is defensive — `find_package(ZLIB REQUIRED)` in
`src/core/data_loaders/CMakeLists.txt` has no vendored fallback (unlike SQLite3,
which tries the system package via `find_package(SQLite3 QUIET)` and falls back to
a vendored amalgamation) — most Linux base images already have it, but it costs
nothing to guarantee.

```bash
module load miniconda/24.4.0-libmamba
./scripts/pipeline/meeting01/gridunesp_setup_env.sh
```

Every later shell (configure, build, and inside the sbatch job) needs:

```bash
module load miniconda/24.4.0-libmamba
conda activate meeting01-build
```

## 3. One-time setup: datasets, configure, build

**Datasets first** (small — 27MB FSDD + 357MB AudioMNIST + 107MB MIT-BIH, well under
the 40TB `/home` quota). The profile (`meeting01-loso.json`) hardcodes absolute
dataset paths for reproducibility — mirror the exact local layout under `$HOME`
instead of forking the profile:

```bash
# from the local machine
scp -r /home/ensismoebius/Documentos/academico/UNESP/doutorado/databases \
  <user>@access2.grid.unesp.br:Documentos/academico/UNESP/doutorado/
```

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

> **Unconfirmed**: `job-nanny`'s documented `INPUT`/`OUTPUT` convention is shown only
> for a handful of named files staged to per-node `/tmp`; this run reads whole
> dataset directories and writes a continuously-growing `results/meeting01/` tree
> over multiple days. The sbatch script sets `SHARED_FS=true` (run directly against
> shared storage, skip local staging) as the best-effort read of the docs — confirm
> with `support.ncc@unesp.br` before a long submission. If wrong, the run is still
> safe to kill and `RESUME=1` restart.

## 5. Bringing results home

Sync back to the local checkout (merges into whatever folds already ran locally —
`RESUME=1` semantics apply on both ends, so this is safe to run mid-flight too):

```bash
rsync -avz <user>@access2.grid.unesp.br:software/nn/results/meeting01/ \
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

## Open questions to confirm with `support.ncc@unesp.br`

- Do **compute** nodes (not the login node) have outbound internet? Affects nothing
  in the plan above (configure runs on the login node specifically to avoid needing
  this), but would simplify things if true.
- Is `SHARED_FS=true` actually the right `job-nanny` setting for a job that reads
  fixed input directories and writes a growing output directory tree, rather than a
  fixed list of named files?
- Realistic queue wait time for `long` (30-day) jobs — affects whether `long` or a
  shorter, resubmitted-via-`RESUME=1` `medium`/`short` chain is the better fit.

## Related

- [Meeting01](../Experiments/Meeting01.md) — the experiment being deployed
- [Build System](./Build-System.md) — `PackageChecking.cmake` / `Vendor*.cmake` design
- [Re-run Runbook](./Re-run-Runbook.md) — local-machine equivalent commands
