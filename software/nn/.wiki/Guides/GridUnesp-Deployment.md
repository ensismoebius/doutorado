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
eval "$(conda shell.bash hook)"
conda activate meeting01-build
```

(The `eval` line matters — see Troubleshooting below for why `conda activate` fails
without it.)

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
./scripts/pipeline/meeting01/gridunesp_deploy.sh
```

Prompts interactively for your GridUnesp username and (unless an SSH key is already
set up) your password, at `ssh`'s own normal password prompt — never captured,
stored, or passed as a script argument. Despite several `ssh`/`rsync` calls inside,
you're only asked once: the first connection is multiplexed and every later call
reuses it. Set `GRIDUNESP_USER=<user>` beforehand to skip the username prompt (e.g.
for a non-interactive/scripted invocation).

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
silent partial dataset). The two PhysioNet sets (eegmmidb, Siena) used to be re-checked file by
file on every run (one HTTPS request each: minutes, even with nothing missing); a finished fetch
now leaves a `.fetch-complete` marker holding the `RECORDS` list, and a later run skips the
per-file pass when `RECORDS` is unchanged and every listed file exists. The marker is removed
before fetching and written only after the last file, so an interrupted run never leaves one.
Run it **on the login node** (confirmed internet; compute-node
internet is unconfirmed, see below) instead of the old "scp a pre-populated tree from
another machine" approach — a fresh GridUnesp checkout no longer depends on the local
machine having downloaded everything first:

```bash
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build   # needs git/wget/sox
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
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build
cmake --preset=max-performance
```

**Build on a compute node**, not the login node — `max-performance` compiles with
`-march=native`, so the binary must be *compiled* on the same CPU model it will
*run* on (Xeon E5-2680 v4), not whatever the login node happens to be:

```bash
srun --partition=short --time=00:30:00 --cpus-per-task=28 --pty bash
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build
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

### Redeploying: what is stamped, what is refused, when it is safe (2026-10-07)

**The problem.** A run lasts weeks, and afterwards the question "which code made these numbers?"
needs an answer that does not depend on anyone's memory. The tree on the cluster has no git
metadata (`~/software/nn/.git` is an empty directory — the deploy copies the working tree, and
the repository root is its parent), and a commit hash alone would be wrong anyway: a deploy
ships uncommitted edits too. So `gridunesp_deploy.sh` works the identity out **locally** and ships
it:

| File on the cluster | Written | Says |
|---|---|---|
| `~/software/nn/SOURCE_REVISION` | by the deploy, **after** the tree | commit, dirty flag and a hash of every source file's content, of the tree that was shipped |
| `~/software/nn/out/build/max-performance/SOURCE_REVISION.built` | by the deploy, only once the remote build **and** the smoke check have succeeded | the revision the binary was built from |
| `~/software/nn/results/meeting01/source_revisions.log` | by the run script, once per start (restarts included) | `2026-10-08T01:02:03Z start resume=0 revision=414e2422+dirty/3fa9c01b2d4e` |

The same label goes into every fold's events file (`session_begin`, `git_commit`) and the live
monitor. The sbatch script runs with `SKIP_BUILD=1`, so `01_meeting01_run_loso.sh` **refuses to
start** unless the first two files are identical. That closes the one way a deploy could lie: a
sync of a newer tree whose build then failed (or was skipped) leaves new sources beside an old
binary, and a job that waited days in the queue would start on it and record the new revision.

```
deploy ok:       SOURCE_REVISION == SOURCE_REVISION.built   -> job starts
build failed:    SOURCE_REVISION  != SOURCE_REVISION.built  -> "REFUSED: the meeting01 binary was
                                                               not built from the sources on disk"
deploy died between tree and revision: no SOURCE_REVISION   -> refused as well
```

What to do, by state of the job:

| Situation | Action |
|---|---|
| job queued (`PD`) or held (`scontrol hold <id>`) | Safe. Redeploy, read the revision the deploy prints, then `scontrol release <id>`; the refusal covers a build that failed. |
| job **running** | **Do not redeploy.** The run script starts the binary anew for every (dataset, fold), so a rebuild underneath it swaps the binary mid-run, while the revision label — read once, at start — keeps naming the old one. *Silent*; nothing guards it. Cancel the job (or wait for it to end), redeploy, then start with `RESUME=1`. |
| `results/meeting01/` holds folds of an older binary | Move them aside first: `mv results/meeting01 results/meeting01_pre_<date> && mkdir -p results/meeting01`. `03_`, `02_` and `04_` refuse folds whose `results_format` is not the current one, and a checkpoint of another format is retrained, never restored — but a directory mixing old and new folds is not a result. |
| `scripts/pipeline/meeting01/.env` exists on the cluster | Delete it: `rm ~/software/nn/scripts/pipeline/meeting01/.env`. Earlier deploys copied your local password file there; nothing on the cluster reads it, and the deploy no longer sends it (it warns if it finds one). |

A queued job also keeps the CPU count and partition it was submitted with — Slurm freezes the
sbatch script at submission — so changing `--cpus-per-task` takes `scancel` and a new `sbatch`,
not a redeploy.

Why a hash of the *content* and not just the commit: see
[Meeting01 § Which code made these numbers?](../Experiments/Meeting01.md#which-code-made-these-numbers-added-2026-10-07).

## Troubleshooting (first real deployment attempt, 2026-09-24)

Four real issues hit in this order during the first actual run of §2/§3 above —
each is concrete, not hypothetical, and each was hit on the real cluster, not the
Docker sim.

### `gridunesp_deploy.sh` must run on the LOCAL machine, not on GridUnesp itself

Running it from an interactive login shell already on `access.grid.unesp.br` fails
immediately:

```
gridunesp_deploy.sh: 'sshpass' not found on PATH -- needed to use the saved
password non-interactively.
```

That's the correct error, not a bug. `sshpass` is a **local-machine-only**
dependency (`_gridunesp_env.sh`'s own header says so) — the script's whole job is
to `ssh`/`rsync` *from* your machine *into* GridUnesp. Running it from inside
GridUnesp means it would try to `ssh` from the login node back to itself, which was
never the design. Run it from your own machine's checkout instead. If you're
already logged into the login node and want to keep going from there anyway, skip
`gridunesp_deploy.sh` and run §3's steps by hand instead — they don't need `sshpass`
because there's no second `ssh` hop involved.

### `conda activate` fails with `CondaError: Run 'conda init' before 'conda activate'`

`module load miniconda/...` only puts the `conda` binary on `PATH` — it does not run
`conda init`, which is what actually installs the `conda activate` shell function
into an interactive login shell's rc file. Without that hook, `conda activate` in
**any** non-interactive shell (an `ssh host bash -s` heredoc, an
`srun ... bash -c "..."` payload, an `sbatch` script) — and potentially an
interactive one too, if `conda init` was never run on this account — falls through
to the raw `conda` binary's own `activate` subcommand, which refuses with exactly
that error instead of doing anything. Fix: source the hook explicitly, once per
shell, right after `module load` and before `conda activate`:

```bash
module load miniconda/24.4.0-libmamba
eval "$(conda shell.bash hook)"
conda activate meeting01-build
```

Every `module load ... && conda activate ...` snippet on this page,
`gridunesp_setup_env.sh`'s own printed instructions, `gridunesp_deploy.sh`'s two
remote blocks, and `01_meeting01_run_loso_gridunesp.sbatch` were all fixed
2026-09-24 to include the `eval` line.

### `tmux`/`screen` fails with `open terminal failed: missing or unsuitable terminal: xterm-kitty`

Only relevant if your local terminal is Kitty. Kitty sets `TERM=xterm-kitty`, and
GridUnesp's terminfo database has no entry for it, so any terminal-aware program on
the remote (`tmux`, `screen`, sometimes `less`/`vim`) fails the same way. Fix, no
install needed — force a `TERM` GridUnesp definitely has before starting the
session:

```bash
TERM=xterm-256color tmux new -s meeting01deploy
```

### Babysitting a long setup across a dropped connection

`ensure_datasets.sh` (24GB, dominated by Siena's 20.3GB) and the compute-node build
can run for hours. If you're driving them from an interactive login-node shell
(§3's by-hand path, e.g. after hitting the `sshpass` issue above) rather than
through `gridunesp_deploy.sh`'s own held-open connection, wrap them in `tmux` **on
the login node itself**, not locally, so the download/build survives your local
connection dropping:

```bash
TERM=xterm-256color tmux new -s meeting01deploy
# inside: module load ...; eval "$(conda shell.bash hook)"; conda activate ...;
# ./scripts/pipeline/meeting01/ensure_datasets.sh; cmake --preset=...;
# srun ... cmake --build ...  (the §3 sequence above, run by hand)
```

Detach with `Ctrl-b d` — the session, and everything running inside it, keeps going
on GridUnesp regardless of your local terminal. Reattach later from the same login
shell with `tmux attach -t meeting01deploy`.

A related, already-fixed robustness gap: `ensure_datasets.sh`'s "already
downloaded" check used to be "does at least one matching file exist" — true for a
genuinely complete download, but also true for one interrupted mid-transfer
(Ctrl-C, a dropped tmux-less SSH session), which would then be silently accepted as
complete on the next run — exactly the "plausible-looking result nobody can trace"
failure mode this project's no-fallbacks rule exists to prevent. Fixed 2026-09-24:
FSDD's clone and AudioMNIST's resample both land in a `.*_staging` directory first
and are only renamed into the path that gets checked once they finish successfully,
so a partial run is never mistaken for a finished one.

### `monitor.py` fails with `SyntaxError: future feature annotations is not defined`

Hit while building the multi-panel `gridunesp_tui.py` control dashboard (below),
and — once checked — found to also silently break the pre-existing
`remote_monitor.sh`, which had never actually been run against the real cluster
end-to-end before this point; its "works unmodified, stdlib only" claim on this
page was wrong until this fix. Cause: GridUnesp's bare `python3` (no module, no
env active) is **3.6.8**. `monitor.py` uses `from __future__ import annotations`
(PEP 563) at module level, which requires Python 3.7+ — under 3.6.8 this is not a
warning, it's a `SyntaxError` before any of the script's own code runs. The
`meeting01-build` conda env did not fix this either: none of its C++-toolchain
packages (openblas, ninja, gcc, ...) pull in a Python interpreter as a
dependency, so `python3` inside the activated env fell through conda's PATH to
the same system 3.6.8. Fixed 2026-09-24: `python=3.11` added directly to
`gridunesp_setup_env.sh`'s package list (pinned, same reasoning as
`gxx_linux-64`/`gcc_linux-64`'s pin). `remote_monitor.sh` updated to `module
load` + activate the env before invoking `monitor.py`, which it previously did
not do at all. Re-run `gridunesp_setup_env.sh` on an env created before
2026-09-24 to pick this up (`conda install`, idempotent, does not recreate the
env).

### `wget` retries a file forever with `HTTP request sent, awaiting response... 416 Requested Range Not Satisfiable`

Hit on eegmmidb, but the same recursive `wget` pattern is used for Siena too. The
original `ensure_datasets.sh` used `wget -c -r ...` (continue + recursive) for both.
`-c` asks for `Range: bytes=<local_size>-` on **every** file, including ones
already fully downloaded from a previous run — and PhysioNet's server correctly
answers a fully-complete file with 416 (there is nothing left in that byte range).
`wget` does not treat that as "already done, move on"; it retries the same URL up
to its default 20 times with growing backoff, which on a directory of ~1500 files
(eegmmidb) is slow and, worse, was observed to hang on this specific loop for
several minutes on one small file (`ANNOTATORS`) before giving up. Fixed
2026-09-24: both fetches now use `-N` (timestamping) instead of `-c`. `-N` compares
the remote `Last-Modified`/size against the local file and skips it outright if
already current — no `Range` request is ever sent for a complete file, so this 416
loop cannot happen — and fetches a missing or stale file fresh (a full re-fetch,
not a byte-range resume, but correct and loop-free). If you hit this before
updating, `Ctrl-C` and re-run `ensure_datasets.sh`; it is idempotent either way.

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
runs against a GridUnesp run unmodified **once the `meeting01-build` conda env has a
real Python** (see Troubleshooting below — GridUnesp's own bare `python3` is 3.6.8,
too old for `monitor.py`'s `from __future__ import annotations`; `python=3.11` was
added to `gridunesp_setup_env.sh`'s package list 2026-09-24 specifically for this).
Two thin wrappers make it easy to reach from the local machine without an
interactive login shell each time:

```bash
# Live view, no local copy of the data -- one SSH session, --plain mode (stdlib
# only, no `rich` needed remotely -- but the conda env IS needed now, for python
# itself, see above). Runs monitor.py's own refresh loop INSIDE that one session
# rather than reconnecting repeatedly.
./scripts/pipeline/meeting01/remote_monitor.sh
./scripts/pipeline/meeting01/remote_monitor.sh --once   # single snapshot
./scripts/pipeline/meeting01/remote_monitor.sh --rank 3

# Or: sync results/meeting01/ down and use the local rich dashboard / archive it
./scripts/pipeline/meeting01/pull_progress.sh
.venv/bin/python3 scripts/pipeline/meeting01/monitor.py --run-tag meeting01_loso
```

Both prompt interactively for your username (and, without a working SSH key, your
password at `ssh`'s own prompt — never captured or stored by either script); set
`GRIDUNESP_USER=<user>` beforehand to skip that prompt. Both also default
`GRIDUNESP_HOST=access.grid.unesp.br` and `GRIDUNESP_REMOTE_DIR=software/nn`
(override via those env vars if your remote layout differs). Both are designed
around the Fail2Ban lockout above: one SSH/rsync connection per invocation, not a
retry loop — use `monitor.py`'s own `--interval` (inside `remote_monitor.sh`'s one
session) or a real-delay `watch -n 60 ...` around `pull_progress.sh` rather than
hammering the login node.

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

> **Results made by a `meeting01` binary older than 2026-10-07 cannot be finished.** Their
> manifests carry no `results_format` (binaries before 2026-10-06) or a 2 (the interim binary
> of commit `b434792e`; the current format is 3), so `03_`, `02_` and `04_` each stop before writing
> anything and name the fold and the reason. There is no repair path. Without a stamp, those
> folds normalized padded windows differently (before commit `6f332734`), left the per-window
> `encoding` empty and wrote `train_ms = 0` for the baselines. With a 2, the LSTM/GRU/Transformer
> frames were a strided gather and the LSTM/GRU cost counted one stack of two. Rebuild
> `meeting01` from the current tree on the cluster (a redeploy does it), then rerun those
> folds: either the whole run without `RESUME` (it clears the checkpoints and reruns every
> fold), or delete only the affected folds' `*_comparative_metrics.csv` and use `RESUME=1` — a
> checkpoint of another `results_format` is retrained, never restored, so no old row survives
> the rerun. Why:
> [Meeting01 § What the Post-Processing Refuses to Read](../Experiments/Meeting01.md#what-the-post-processing-refuses-to-read-found--fixed-2026-10-06-second-pass).

## 6. Optional: all-in-one control TUI

`scripts/pipeline/meeting01/gridunesp_tui.py` is a multi-panel terminal dashboard
that wraps everything in §2–§5 into one screen instead of running each script by
hand: dataset-fetch progress, build state, the Slurm queue, and live training
progress, all updating together from **one** persistent SSH session (added
2026-09-24). It is a **controller, not a reimplementation** — every panel's data
comes from `gridunesp_status_remote.py` (which itself reuses `monitor.py`'s own
`SessionState`/`EventTailer` classes for the training numbers, not a second,
parallel readout of the same event files), and every action key shells out to the
same scripts documented above:

```bash
.venv/bin/python3 scripts/pipeline/meeting01/gridunesp_tui.py
.venv/bin/python3 scripts/pipeline/meeting01/gridunesp_tui.py --interval 30
```

| Key | Action | Runs |
|---|---|---|
| `d` | Deploy | `gridunesp_deploy.sh` (§2/§3) |
| `s` | Submit | the same `sbatch ...` call as §4, behind a confirm dialog (this starts a job that can run up to 30 days) |
| `m` | Full monitor | `remote_monitor.sh` (§4) — the complete `rich` training dashboard, fullscreen |
| `p` | Pull results | `pull_progress.sh` (§5) |
| `v` | Validate toolchain | `run_gridunesp_docker_sim.sh` (§2, local Docker sim) |
| `r` | Reconnect | manual only — a dropped connection is never auto-retried (Fail2Ban, same reasoning as everywhere else on this page) |
| `q` | Quit | — |

`d`/`m`/`v` use Textual's `App.suspend()`: the dashboard steps aside, the other
script gets the real terminal (so `gridunesp_deploy.sh`'s own prompts and
`remote_monitor.sh`'s own fullscreen dashboard work exactly as they do run
directly), and the control TUI resumes when it exits.

**LOCAL machine only** — `textual` (`scripts/requirements.txt`) is never needed on
GridUnesp itself; the remote-side script it drives is stdlib + `monitor.py`'s
ingestion classes only. Needs the SAME `sshpass` + `.env` credentials as every
other script here (shared file, `scripts/pipeline/meeting01/.env` — first run
prompts once, same as `gridunesp_deploy.sh`).

## 7. Remote GA collection from the dashboard

The web dashboard (see [Meeting01 § Web dashboard](../Experiments/Meeting01.md#web-dashboard-fastapi--plotlyjs))
can fetch GA search results directly from GridUnesp via SSH, without manually
syncing `results/meeting01/` first.

### How it works

1. **Remote side:** the command that `collect_ga_local.py` sends over SSH does
   `cd <remote_dir>`, activates the remote conda env, then pipes a small Python
   snippet on **stdin** (a quoted heredoc, not `python3 -c`) that imports
   `collect_ga()` from `gridunesp_status_remote.py` and prints its result as one
   JSON object. The activation must be, in order:

   ```bash
   module load miniconda/24.4.0-libmamba \
     && eval "$(conda shell.bash hook)" \
     && conda activate meeting01-build
   ```

   Skipping the `eval` hook makes `conda activate` fail outright; skipping the
   whole chain silently falls back to the login node's `python3` (3.6.8), which
   cannot even parse `from __future__ import annotations`. The `&&` chaining is
   load-bearing for exactly that reason.
2. **Local side:** `collect_ga_local.py` writes the payload to
   `<results_dir>/<run_tag>_ga_remote.jsonl`, one JSON object per line, with
   per-cell fields (`n_individuals`, `n_generations`, `best_val_mse`,
   `best_inference_cost`) plus `ts` and `source`.
3. **Dashboard:** `POST /api/ga/remote/collect` performs that collection
   synchronously. `GET /api/ga/remote` reads the last **complete** record from
   the JSONL and returns it as JSON.

**Parsing is deliberately tolerant.** Cluster SSH prepends a fixed banner to
every connection (an OpenSSH post-quantum-key-exchange warning plus a `====`
block around the `module load` conda notice), so `collect_ga_local.py` scans
stdout **backwards** for the last line that parses as a JSON object. Error
reporting strips the same boilerplate and keeps the tail, because the actual
cause is a Python traceback at the very end — a naive head-truncation shows
only the banner and hides the failure.

**Quoting.** Each interpolated value gets exactly one quoting context.
`GRIDUNESP_REMOTE_DIR` lands in shell position (`cd <dir>`) and is
`shlex.quote`d; the results path and run tag are embedded in the Python body
with `repr()`. Running `shlex.quote` over a value that `repr()` then re-quotes
yields a string with literal quote characters embedded in it — a silently wrong
path rather than a loud failure. Because the heredoc delimiter is quoted
(`<<'EOF'`), the shell does no expansion inside the Python body, so a
caller-supplied `results_dir` is shell-inert.

### Setup

Credentials are read from `scripts/pipeline/meeting01/.env` (shared with
`gridunesp_deploy.sh`). Variables:

```
GRIDUNESP_USER=<your-username>            # required
GRIDUNESP_PASSWORD=<your-password>        # optional — omit when using an SSH key
GRIDUNESP_HOST=access.grid.unesp.br       # default
GRIDUNESP_PORT=22                         # default
GRIDUNESP_REMOTE_DIR=software/nn          # default
GRIDUNESP_CONDA_ENV=meeting01-build       # default
```

Keep the file mode at `600`; it may hold a cluster password.

Two **mutually exclusive** authentication modes, selected by whether
`GRIDUNESP_PASSWORD` is set:

| Mode | Trigger | Command shape |
|---|---|---|
| Password | `GRIDUNESP_PASSWORD` set | `sshpass -e ssh …` with `SSHPASS` exported into the child environment |
| SSH key | password unset | `ssh -o BatchMode=yes …` |

`BatchMode=yes` disables all interactive prompts, which is what makes the
dashboard's HTTP request safe to serve (a request must never block on a terminal
prompt), and it is also why key auth must be set up beforehand: without a key and
without a password there is no way in, by design. The password is passed to
`sshpass` via its `-e` flag and the `SSHPASS` environment variable rather than on
the command line, so it does not appear in the remote process's `argv`.

### Usage

```bash
# one-shot collection (writes <run_tag>_ga_remote.jsonl)
.venv/bin/python scripts/pipeline/meeting01/collect_ga_local.py

# continuous collection every 60s
.venv/bin/python scripts/pipeline/meeting01/collect_ga_local.py --interval 60

# print the remote command without running SSH (debugging aid)
.venv/bin/python scripts/pipeline/meeting01/collect_ga_local.py --print-cmd
```

Or trigger from the dashboard: open the **Architecture Search** tab and click
**Collect from GridUnesp**. The button shows status (collecting / done / error)
and populates the GA panel with remote results. The endpoint is synchronous and
bounded by `SSH_TIMEOUT_S` (60 s).

> **Prerequisite — the remote checkout must contain `collect_ga`.** The remote
> command imports `collect_ga` from
> `scripts/pipeline/meeting01/gridunesp_status_remote.py` **on the cluster**. A
> cluster checkout predating that function fails with
> `ImportError: cannot import name 'collect_ga'`, which the collector surfaces
> verbatim after stripping the SSH banner. Check the remote tree before relying
> on the button:
>
> ```bash
> .venv/bin/python scripts/pipeline/meeting01/collect_ga_local.py --print-cmd
> # then, on the cluster:
> grep -c '^def collect_ga' ~/software/nn/scripts/pipeline/meeting01/gridunesp_status_remote.py
> ```
>
> Note that a checkout made by `rsync`/`scp` (rather than `git clone`) has no
> git history, so there is no `git rev-parse HEAD` to compare against — compare
> the file contents instead. Syncing a new file over a checkout that a running
> job is using can disturb that job, so treat any remote update as an
> intentional, separately-verified step.

### SSH key setup (recommended)

To avoid re-entering your password and to keep `BatchMode=yes` viable, set up an
SSH key:

```bash
# on your local machine
ssh-keygen -t ed25519 -f ~/.ssh/gridunesp -N ""
ssh-copy-id -i ~/.ssh/gridunesp.pub <user>@access.grid.unesp.br
```

Then leave `GRIDUNESP_PASSWORD` unset in `.env`; `collect_ga_local.py` uses the
key with `BatchMode=yes` and never prompts.

> **Note:** GridUnesp's Fail2Ban triggers after rapid repeated login attempts.
> The `POST /api/ga/remote/collect` endpoint runs one SSH session per call —
> avoid hammering it, and run the server single-worker so a burst of clicks
> cannot fan out into concurrent sessions. The one-shot `collect_ga_local.py` is
> designed around this: one connection per invocation, not a retry loop.

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
