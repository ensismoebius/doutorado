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

## Forward-looking: other experiments use a different dataset mechanism

This runbook covers `meeting01` only, and `meeting01` never touches SQLite — its profile
(`meeting01-loso.json`) hardcodes absolute WAV-directory paths
(`/home/ensismoebius/.../databases/{fsdDataset,audioMNIST_8k,mitbih}`), which is exactly
why §3 above mirrors the whole `databases/` tree to the *same absolute path* under the
grid account's `$HOME` rather than doing anything sqlite-specific.

`thesis`, `paraconsistentGA`, and `autoencoderRunner` are a different story: they all read
a single `~/database.sqlite` file (three independent call sites —
`ThesisDataset::load_dataset`, `SqliteBatchSource`, `TrialFoldSelector` — each opens its
own `sqlite3` connection, but all resolve the path the same way, via
`nn::utility::expand_home("~/database.sqlite")`). None of these are part of the current
GridUnesp plan. If one of them is submitted to the grid later, the fix is one line, not a
code change: `scp` the local `database.sqlite` to `~/database.sqlite` on
`access2.grid.unesp.br` once, and `expand_home()` picks it up automatically on every
later run, the same as any other `~`-relative path on that account.

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
