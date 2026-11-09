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
| job **running** | **Do not redeploy — `gridunesp_deploy.sh` refuses (since 2026-10-13).** The job stages only the `meeting01` executable; every (dataset, fold) is a new process that loads `libstatistics.so`, `libmatioCpp.so` and `libmatio.so` from the build tree (the binary's RPATH is absolute), so a rebuild underneath it changes the code of its later folds, while the revision label — logged once, at start — keeps naming the old one. That was *silent* until the deploy started checking `squeue` for a running `meeting01-loso*` job. Stop it (`scancel --batch --signal=USR1 <id>` copies its results back, see [Stopping](#stopping-a-run-the-30-day-limit-scancel-and-why-slurms-own-signal-never-reaches-it-2026-10-13)) or wait for it to end, redeploy, then start with `RESUME=1`. |
| `results/meeting01/` holds folds of an older binary | Move them aside first: `mv results/meeting01 results/meeting01_pre_<date> && mkdir -p results/meeting01`. `03_`, `02_` and `04_` refuse folds whose `results_format` is not the current one, and an interrupted GA search of another format or profile is refused on resume — but a directory mixing old and new folds is not a result. |
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
`01_meeting01_run_loso.sh` under `job-nanny` (GridUnesp's mandatory job wrapper — how it is
used here, and why the first submission died in one second, is in
[job-nanny below](#job-nanny-what-it-really-does-and-how-this-job-uses-it-2026-10-13)).
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
local-machine behavior (see [Meeting01](../Experiments/Meeting01.md#running-it)). That file is
the last thing a fold writes, through a temporary file and a rename (since 2026-10-13), so it
exists only once the fold has written everything — even for a run stopped in the middle of a
write; why that matters here is in
[Stopping](#stopping-a-run-the-30-day-limit-scancel-and-why-slurms-own-signal-never-reaches-it-2026-10-13).

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

### job-nanny: what it really does, and how this job uses it (2026-10-13)

**What went wrong first.** Job 5245836 waited five days in `long`, started on node036 and died
in one second: `job-nanny: command not found` (exit 127). The script had been written from the
docs and never run. The callout that stood here even said to smoke-test it in `short` first; nobody
did.

**Where job-nanny lives.** The v3 docs ("Melhorando o Script de Submissão"): *"O script
`job-nanny` é reconhecido ao carregar o módulo **gridunesp**. Quando o usuário faz login no
cluster, esse módulo é carregado automaticamente."* A batch job is not a login shell, so the
module is not loaded there. `module show gridunesp` puts `/opt/gridunesp/dist/gridunesp/bin` on
`PATH`, where `job-nanny` is a symlink to `goo-job-nanny-v2`, a 131-line bash script. The sbatch
script now runs `module load gridunesp` first.

**What it does — read from its source, not from the docs:**

| Step | `goo-job-nanny-v2` | Consequence here |
|---|---|---|
| start | exits 3 if `INPUT` or `OUTPUT` is unset | both are required, not optional metadata |
| work dir | `/store/$USER/$SLURM_JOB_ID/` when `SHARED_FS=true`, else node-local `/tmp/…` | `/store`: no automatic cleanup, survives a node failure |
| copy in | `rsync -a -u $INPUT $WORK_DIR`, **no `-R`** | `out/build/…/meeting01` would arrive as plain `meeting01`, so `INPUT` is one directory already laid out like the tree |
| run | `cd $WORK_DIR`, then the command | the run's root is the copy, not `~/software/nn` |
| during | every `WAIT_CHECKPOINT` s: `rsync -a -u $OUTPUT <submit dir>` | results appear in `~/software/nn/results` every 30 min (set to 1800 s); nothing is ever deleted there, so a file the run deletes later stays ([stale copies](#files-the-run-deletes-come-back-the-stale-copy-hazard-2026-11-09)) |
| end | copies `OUTPUT` back, deletes the work dir, exits with the command's own code — only when the command returns | a failed run is a failed job |
| killed | no signal handling at all: no final copy, no cleanup | `~/software/nn` has the last periodic copy; the newest state stays in `/store/$USER/<jobid>/`. The sbatch script makes sure the command *returns* first — see [Stopping](#stopping-a-run-the-30-day-limit-scancel-and-why-slurms-own-signal-never-reaches-it-2026-10-13) |

**How the sbatch script feeds it.** It builds `.job-input-<jobid>/` in `~/software/nn` with
exactly what the run reads, in the same layout: `SOURCE_REVISION`, `scripts/pipeline/meeting01/`,
the profile, the binary, its `.built` stamp, a small `start.sh`, and — for `RESUME=1` only —
`results/meeting01/`. That directory is `INPUT`; its `results/` is `OUTPUT`, which lands on
`~/software/nn/results`. job-nanny's command is `start.sh`: by the time it runs, job-nanny has
copied the stage to `/store` and only that copy is used, so it deletes the one in `~/software/nn`
at once, then starts the run in a process group of its own (why: Stopping, below).

```
~/software/nn/.job-input-<id>/   --INPUT-->   /store/<user>/<id>/.job-input-<id>/   (run here)
   deleted by start.sh seconds after the start
~/software/nn/results/          <--OUTPUT--   /store/<user>/<id>/.job-input-<id>/results/
                                              (every 30 min, and at the end)
```

Until 2026-10-13 the stage was meant to be deleted by the script's EXIT trap "when the job
ends, also on TIMEOUT". It was not: smoke job 5259424 left `.job-input-5259424/` behind, for the
reason the next section explains.

#### Stopping a run: the 30-day limit, `scancel`, and why Slurm's own signal never reaches it (2026-10-13)

**The problem.** `long` allows 30 days and the full LOSO grid needs more (weeks to months, per
the profile's `_total_runs_breakdown`), so the first job will most likely end *at its time
limit*, and a `RESUME=1` job will continue from whatever that ending left behind. The ending
at the limit is the main path, not an edge case.

**What the limit did to smoke job 5259424** (limit 15 min, `KillWait` = 120 s on this cluster):

```
03:42:46  limit     slurmstepd: "*** JOB 5259424 ... CANCELLED AT 03:42:47 DUE TO TIME LIMIT ***"
                    SIGTERM -> the batch shell only
03:42:47..03:44:47  the run went on training; job-nanny made one more periodic copy
03:44:47  +120 s    SIGKILL -> every process of the job

sacct:  5259424         TIMEOUT    00:15:09   0:0
        5259424.batch   CANCELLED  00:17:10   0:9      <- died of signal 9, 120 s after the limit
left:   ~/software/nn/.job-input-5259424/        /store/ensismoebius/5259424/
```

Two facts combine:

1. **The SIGTERM reaches the batch shell, not the processes under it.** With
   `ProctrackType=proctrack/cgroup` (this cluster's setting) Slurm signals the step's own task —
   for a batch job, the shell running the sbatch script — and its descendants get only the
   SIGKILL that follows `KillWait`. Both halves are measured: the shell does get it (smoke job
   5259471, below — its handler ran in the second of the `scancel`), and the processes under it
   do not (here: job-nanny, a plain bash script with no handler, would otherwise have died at
   once, and the shell's deferred trap would then have run).
2. **A shell running a foreground command defers its traps until the command returns.** The
   batch shell was running `job-nanny …` in the foreground, so its own trap waited behind a
   command that was never going to return before the SIGKILL.

So nobody stopped the run: job-nanny's finish — final copy back, deleting its work dir — only
happens when its command returns, and the shell's trap only when job-nanny returns.

**What the sbatch script does now** — the shell stays free to act, and acts before the limit:

```
limit - 15 min   Slurm: USR1 -> batch shell                     (#SBATCH --signal=B:USR1@900)
                 the shell is in `wait` (job-nanny runs in the background), which a trapped
                 signal interrupts at once -> stop_run:
                     SIGTERM -> the run's process group    (start.sh started the run with setsid
                                                            and wrote the group id to
                                                            /tmp/meeting01-run-<id>.pgid)
                 run script + meeting01 binary die -> job-nanny's command returns 143 ->
                 job-nanny's normal finish: final copy back, rm -rf /store/<user>/<id>/
                 shell: "[gridunesp] STOPPED ... Continue with: RESUME=1 sbatch ...", exit 143
limit            nothing left to kill
```

| at the time limit | before (job 5259424) | now |
|---|---|---|
| who stops the run | nobody — SIGKILL at limit + 120 s | the batch shell, 15 min before the limit |
| results in `~/software/nn` | the last periodic copy (up to 30 min old) | all of them — job-nanny's final copy ran |
| `/store/<user>/<id>/` | left behind; clean by hand | deleted by job-nanny |
| `~/software/nn/.job-input-<id>/` | left behind | deleted by `start.sh` seconds after the start |
| job state | `TIMEOUT` | `FAILED`, sacct ExitCode `15:0`, a `STOPPED` line in the `.err` |

`15:0` is how this cluster's Slurm (22.05) shows the script's `exit 143`: a probe job whose whole
script was `exit 143` came out the same, `FAILED 15:0` (job 5259470, 2026-10-13). Read it as
"128 + 15, SIGTERM", not as a code the run chose.

The same stop on demand: **`scancel --batch --signal=USR1 <jobid>`**. `--batch` matters — without
it `scancel --signal` goes to job steps, and this job has none (everything runs inside the batch
step). A plain `scancel <jobid>` sends SIGTERM to the batch shell, which runs the same handler,
but then only the 120 s of `KillWait` remain for the final copy before the SIGKILL. Measured on
smoke job 5259471: `scancel` at 04:22:47, the handler's `stop signal` line in the same second,
final copy and `/store` cleanup done, the batch step ended 1 s later (sacct: job `CANCELLED`,
batch step `FAILED 15:0`).

**Why a run stopped at an arbitrary instant cannot corrupt what `RESUME=1` reads.** `RESUME=1`
skips a fold whose `*_comparative_metrics.csv` exists and reruns the others; since 2026-11-02 it
restores no finished run of a fold (see [Meeting01 § Cross-review](../Experiments/Meeting01.md#cross-review-with-a-second-agent-2026-11-02)),
only the search cut mid-way, whose state is written to `<file>.tmp` and renamed (an interrupted
write leaves a stray `.tmp`, never a torn state) and whose cache drops a torn last line. That marker was **not** written that way until 2026-10-13:
the binary wrote it straight to its final path, and *first* of the fold's final outputs. A
process killed during that write left a truncated, non-empty file — skipped by the next
`RESUME=1` as a finished fold, read by `02_` as one. **Silent**, and newly reachable now that
stopping the run at any moment is routine. It now goes through `.tmp` + rename and is written
*last*, after the publication table, summary and LaTeX exports (`write_rows_csv`,
`write_experiment_outputs`; tests `DATWriterTest.ComparativeCsv*`), so "the file exists" means
"the fold wrote everything".

**What can still go wrong, and how it shows:**

- A signal during the setup (before job-nanny starts) ends the job without starting the run:
  `stop signal during setup -- not starting the run`, exit 143, the stage removed by the EXIT
  trap. After job-nanny starts, `stop_run` waits up to 30 s for the process-group file (job-nanny
  may still be copying the stage in). If it never appears, the handler says so in the `.err` and
  stops nothing; the limit then plays out as it did for 5259424 — visible: no `STOPPED` line,
  `/store/$USER/<id>/` still there.
- Any SIGKILL before the final copy finishes (a node failure; a plain `scancel` whose final
  copy takes over 120 s) leaves the same two things behind.

Recovery in both cases (from `~/software/nn`; the listing step is the
[reconcile](#files-the-run-deletes-come-back-the-stale-copy-hazard-2026-11-09) done by hand, because
this `rsync` does not delete either):

```bash
source scripts/pipeline/meeting01/_job_nanny_reconcile.sh
job_nanny_write_listing /store/$USER/<id>/.job-input-<id>/results <id>
rsync -a /store/$USER/<id>/.job-input-<id>/results ~/software/nn/
job_nanny_reconcile results results/meeting01 <id>          # an array task: add its two globs, as the sbatch does
rm -rf /store/$USER/<id> ~/software/nn/.job-input-<id>      # then RESUME=1 sbatch …
```

**A fresh run refuses a non-empty `results/meeting01`.** Before job-nanny the run script deleted
old checkpoints in place. Now it deletes them inside the work dir, and the copy back never deletes
anything, so old checkpoints would sit next to the new rows. Move the old directory aside, or
continue it with `RESUME=1`.

#### Files the run deletes come back: the stale-copy hazard (2026-11-09)

**The problem.** A GA search writes two resume files while it runs,
`meeting01_ga_<tag>_checkpoint.json` (the population after each generation) and `_cache.jsonl`
(every genome scored so far), and deletes both when it finishes: a finished search has nothing to
resume. job-nanny's copies use `rsync -a -u` without `--delete`, so a copy made *while* the
search ran puts the two files in `~/software/nn`, and nothing ever takes them away:

```
time ─►        gen 1 saved      periodic copy     gen 2, search done      final copy
/store         ckpt(gen 1)      ──────────────►   ckpt deleted            ──────────────►  (no ckpt)
~/software/nn                   ckpt(gen 1)                                                ckpt(gen 1)   ← stale
```

Measured: pre-validation job 5317827 (2026-11-07, exit 0, every search finished) left 13
checkpoints and 13 caches at home. Each checkpoint is older than its own search's selection
manifest; four stopped at generation 0 or 1 of 2.

**Why it matters, and why nobody would notice.** `RESUME=1` stages `results/` *from home*. In a
fold that is rerun, 01_meeting01_run_loso.sh promises that every finished run is recomputed and
only the search that was cut mid-way continues from its checkpoint. With a stale checkpoint a
search that had *finished* is "continued" from generation 1 instead:

| | the search was cut mid-way | the search had finished (stale copy) |
|---|---|---|
| checkpoint at home | the newest state, from the final copy | an old generation, from a periodic copy |
| what RESUME does | continues it — intended | continues it — not intended: it should start over |
| stamp check (config hash, results_format) | passes | **passes too**: same configuration |
| what the events say | "resumed" — true | "resumed" — false; the earlier attempt's events stay beside the new ones |

Silent: nothing fails, and with a deterministic search the final number may even come out the
same, but the run's record says a search was interrupted that never was.

**The fix** (`scripts/pipeline/meeting01/_job_nanny_reconcile.sh`, used by both sbatch files):

1. `start.sh` no longer `exec`s the run. It waits for it — a run ended by `stop_run` included —
   and writes `results/.job-final-files-<id>.txt`: every file the run left, nothing it deleted.
   job-nanny's final copy brings that list home.
2. After job-nanny returns, the sbatch removes from `$RES` every file the list does not name and
   that is not newer than the list (every stale copy is older: `rsync -a` keeps the run's
   mtimes), prints each removal, and deletes the list. An array task looks only at its own fold's
   names (`meeting01_loso_<ds>_fold<f>_*`, `meeting01_ga_meeting01_loso_<ds>_fold<f>_*`): the 24
   tasks share `results/meeting01`, and one task's list says nothing about the others' files.
3. No list (job-nanny SIGKILLed before its final copy): nothing is removed and the `.err` says
   `NOT RECONCILED`. Recover from `/store` as above.

Tests: `bash scripts/pipeline/meeting01/_job_nanny_reconcile.sh --self-test` (scope, globs, a
spaced name, a file newer than the list, a missing or foreign list); the two `start.sh` heredocs
were run in a fake work dir (exit code passed through, list written also after a TERM to the run's
process group). Not yet run on the cluster: the next smoke job should end with
`[reconcile] job <id>: N stale file(s) removed`.

**What still touches `/home` during the run** (the docs say jobs doing heavy `/home` I/O may be
cancelled): the datasets, read from their absolute paths in the profile once per fold; the
shared libraries `libstatistics.so`, `libmatioCpp.so` and `libmatio.so`, which every fold's
process loads from the build tree (the binary's RPATH is absolute; only the executable is
staged) — the reason `gridunesp_deploy.sh` refuses to run while a `meeting01-loso*` job does.
(`latex_data_dir` used to write a few small files per fold under
`~/Repos/doutorado/documentation/…` too — files nothing read, the paper's data all coming from
`02_`/`04_`, run locally (§5). It is empty since 2026-11-02.)

**Smoke test — run it after any change to this path** (~6 min in `short`; it stops itself
before its limit on purpose — that is the ending the real job will have):

```bash
WAIT_CHECKPOINT=60 sbatch -p short -t 00:06:00 -c 4 --signal=B:USR1@150 \
  --job-name=meeting01-loso-smoke scripts/pipeline/meeting01/01_meeting01_run_loso_gridunesp.sbatch
# meeting01_loso_<id>.err should end with: "stop signal: ending the run", job-nanny's
#   "Return Code 143" and "Removing files from /store/...", then "[gridunesp] STOPPED ..."
# gone: ~/software/nn/.job-input-<id>/ (seconds after the start) and /store/$USER/<id>/
# afterwards: mv results/meeting01 ~/meeting01_smoke_<id>   (a fresh run refuses a non-empty one)
```

`--signal` on the command line replaces the script's own `@900`, and Slurm may deliver it up to a
minute early. **Any `-t` under 15 minutes needs its own `--signal`**: Slurm sends the signal once
the job is within that many seconds of its end, so with `@900` on a 6-minute job it comes at the
start, and the job ends with `stop signal during setup -- not starting the run`. Keep the
`meeting01-loso` prefix in any `--job-name`: it is how the deploy recognises a running job.

Job 5259424 (2026-10-13, the earlier 15-minute version of this test) showed the start path
works: the module loaded, the staging reached `/store`, the run printed revision
`1977723e/b87ea04341e1`, built fold 0's split, passed the leakage gate and began training, and
the periodic copy put the events file and `source_revisions.log` in
`~/software/nn/results/meeting01`. Its *ending* is what [Stopping](#stopping-a-run-the-30-day-limit-scancel-and-why-slurms-own-signal-never-reaches-it-2026-10-13)
is about.

The current version passed on 2026-10-13, at commit `1f9003e6`:

| job | ended by | what happened |
|---|---|---|
| 5259463 | `--signal=B:USR1@150` on a 6-min limit | USR1 at 3:22; run stopped (job-nanny: `Return Code 143`), final copy, `/store/…/5259463/` deleted, `STOPPED` line; job `FAILED 15:0` at 3:27 |
| 5259471 | plain `scancel` after 1 min | same sequence, 1 s from `scancel` to the end of the batch step; job `CANCELLED` |
| both | — | `.job-input-<id>/` already gone 20 s after the start; results, events, checkpoints and `source_revisions.log` (`1f9003e6/166ec23c148f`) in `~/software/nn/results/meeting01` |

While 5259471 ran, `gridunesp_deploy.sh` refused, naming it (`5259471 meeting01-loso-smoke`),
before sending anything.

### Threads: why 20 CPUs made the run about 1000× slower (2026-10-23)

**The problem.** The first real job (5259474, 20 CPUs, `OMP_NUM_THREADS=20`) ran for 6 days 17 hours
and finished 3 epochs of the first of 8 runs of the first of 24 (dataset, fold) pairs. An epoch
took about 2 days. The same epoch takes about 80 seconds on one core of a desktop. **Silent**: the
job was `RUNNING`, its CPUs were busy (Slurm's `AveCPU` showed ~9.5 of 20 cores in use), and the
only sign was the time between `epoch N/30` lines in the `.err` log.

**Why more threads cannot help here.** One training step is one window: `samples_per_batch = 1`.
The LSTM then does 512 sequential steps, each a matrix times a single vector. For hidden size 64
(the size the cost formulas in [Meeting01](../Experiments/Meeting01.md) use as their example; the
architecture search picks the real one) that is about 18,000 multiply-adds per call — too little to
split across threads, so every call pays the cost of waking and joining 20 threads and gets
nothing back.

```
one window, one training step (batch size 1), hidden size 64

  step t  : gates = W · [x_t, h_{t-1}]        <- (4·64) x (8+64) times a vector: 18,432 MACs
  step t+1: ...                                  cannot start before step t has finished
  ... 512 steps in a row, and a backward pass of the same length

  1 thread : 512 small multiplies, one after the other
  20 threads: 512 small multiplies, each preceded and followed by a 20-thread rendezvous
```

**Measured locally** (same fold, seed and caps as the production profile, 12-core desktop; seconds
per training step, first LSTM-AE run, latency encoding):

| `OMP_NUM_THREADS` | 1 | 2 | 4 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|
| s / step | **0.41** | 0.58 | 0.62 | 0.81 | 1.03 | **9.2** |

More threads is monotonically slower, and past the number of cores (12 here) it is catastrophic:
20 threads cost 22× the single thread. The cluster job asked for 20 CPUs, but those are
hyperthreads — 10 physical cores — and its epoch times were far worse than this table would predict:

```
cluster, 20 threads        steps  1..21 : 2.7 s each
                           steps 23.. : ~1,200 s each, for 6+ days   (events file, epoch_progress)
local, 20 threads / 12 cores         : 9.2 s each
local,  1 thread                     : 0.41 s each
```

What is **measured**: more threads is slower, locally, at every count I tried; the cluster's
step-time jumped from 2.7 s to ~1,200 s and stayed there. What is **not** established: why the
cluster jumped by ~440× at one particular step, which local oversubscription (22×) does not
reproduce. The single-thread cluster smoke test below is the test of that.

**Same tensors, same maths.** Epoch 1 gives train 1.000119 / val 1.000270 locally and 1.000119 /
1.000271 on the cluster, so the slowdown is the environment, not different data or code.

**What the sbatch now does.** `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` both follow
`--cpus-per-task`; the thread count is therefore chosen in exactly one place, the CPU request.
OpenBLAS sizes its own pool separately from OpenMP, so setting only the OpenMP variable left it
free to multiply the two. Since 2026-11-02 the sbatch's own default is `--cpus-per-task=1` (it
said 20, a choice made before this measurement; `submit_loso_array.sh` passed `-c 1` anyway,
which overrides it), and `01_meeting01_run_loso.sh` run locally exports both variables as 1
unless the caller set them, and prints them: unset, each library sized its pool from the
machine, the slow side of the table above.

**Not a thread problem: the loss.** At every epoch, locally and on the cluster, the loss sat at
1.0001, which is the variance of the z-scored windows — the model outputs the mean and has learnt
nothing. See [Meeting01](../Experiments/Meeting01.md) for whether it ever leaves that value.

### Job array: one task per (dataset, fold) (2026-10-24)

**Problem.** One 20-CPU job runs the 24 (dataset, fold) pairs in series, and the program is
~1000× slower with 20 threads than with 1 (previous section). One-CPU jobs that run side by side
avoid the thread penalty *and* a single failure no longer stops everything.

```
task t   0..5   fsdd fold 0..5        task t = dataset[t / 6], fold t % 6
         6..11  audiomnist             one Slurm job per task, 1 CPU, its own /store work dir
         12..17 eegmmidb               and its own job-nanny; results meet in results/meeting01
         18..23 siena
```

Use `scripts/pipeline/meeting01/submit_loso_array.sh` (from `~/software/nn`):

| You want | Command |
|---|---|
| see what would be submitted | `submit_loso_array.sh --dry-run` |
| submit the unfinished folds | `submit_loso_array.sh` |
| recover after a limit / `scancel` / node failure | the same command — a fold with its `*_comparative_metrics.csv` is skipped, a cut-short one is rerun, continuing only its interrupted search |
| a smoke test | `--partition short --time 00:06:00 --extra "--signal=B:USR1@150 --job-name=meeting01-loso-smoke" --only "0 6"` |

What it enforces (each a *loud* failure):

* **No double submission:** refuses while any `meeting01-loso*` job is RUNNING or PENDING.
* **The mapping cannot drift:** each task checks the profile still has the same 4 datasets and 6
  folds, else it refuses (edit `ARRAY_DATASETS` / `ARRAY_FOLDS` in the sbatch).
* **Fresh task, existing files:** a task that is not `RESUME` refuses if *its own* fold already has
  files. (The whole-directory check of a single job cannot be used: other tasks copy results into
  the same directory while a late task is still queued.)
* **Provenance:** each task writes `source_revisions_<arrayjob>_<task>.log`. A shared log would
  lose lines: job-nanny copies the whole directory with `rsync -a -u`, and the newest copy wins.
* **Deploy guard unchanged:** a redeploy is refused while *any* task is RUNNING, so no fold runs
  on a half-replaced build tree. A fully PENDING array may be redeployed; every task logs the
  revision it starts with.

**Not measured:** the wall time of a real fold at 1 thread. The profile's `_total_runs_breakdown`
was calibrated with the 512-frame baselines and is now too high for them.

### The baseline pre-validation on the grid (2026-11-02)

**What it is.** The evidence for removing LSTM-AE, GRU-AE and Transformer-AE from the grid: every
family through one reduced search on the development folds, 8 one-thread processes
([Meeting01 § Sequence baselines removed](../Experiments/Meeting01.md#sequence-baselines-removed-and-measured-once-2026-11-02)).
`00_meeting01_prevalidate_gridunesp.sbatch` runs `00_meeting01_prevalidate_baselines.sh` under
job-nanny with the same stop handler and staging as the grid's sbatch.

**Why one job, not an array.** The result is a *ratio* of wall times — each family's fold over
the SNN-AE's. Array tasks land on different nodes beside different co-tenants, so each family
would be timed under its own conditions:

```
array (rejected)                       one job (used)
  node A: snn-ae  fsdd   + strangers     node X, 16 CPUs (8 cores x 2 hyperthreads):
  node B: lstm-ae fsdd   + others          snn-ae  fsdd | lstm-ae fsdd | gru-ae fsdd | tf-ae fsdd
  ...      ratio = two different nodes     snn-ae  eeg  | lstm-ae eeg  | gru-ae eeg  | tf-ae eeg
                                           ratio = same node, same neighbours, a core each
```

16 CPUs for 8 processes because a CPU here is a hyperthread: with 8, two processes could share a
core, a slowdown that would hit some families and not others.

**Interrupted tasks start over.** The binary continues a cut search from its GA checkpoint and
truncates the events log at the restart, so a resumed task's `fold_begin → fold_end` would time
only the part after the restart — a shorter fold, **silently**. Under `RESUME=1` the runner moves
an unfinished task's files to `<family>/interrupted_<time>_<ds>_fold<f>/` and reruns it whole.
Hence the generous limit (`medium`, 5 days) for a run estimated at under a day.

```bash
# smoke (tiny profile copied to ~/prevalidation_smoke/, ~10 min in short)
PREVALIDATION_PROFILE=$HOME/prevalidation_smoke/smoke_profile.json \
PREVALIDATION_DIR=results/meeting01_prevalidation_smoke \
sbatch -p short -t 00:20:00 --signal=B:USR1@150 --job-name=meeting01-loso-preval-smoke \
  scripts/pipeline/meeting01/00_meeting01_prevalidate_gridunesp.sbatch
# the real run
sbatch scripts/pipeline/meeting01/00_meeting01_prevalidate_gridunesp.sbatch
```

**Afterwards, locally** (03_ and the report write the paper's files; the datasets are local too):

```bash
rsync -avz <user>@access.grid.unesp.br:software/nn/results/meeting01_prevalidation/ \
  results/meeting01_prevalidation/
.venv/bin/python3 scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py \
  --results-dir results/meeting01_prevalidation/snn-ae --run-tag meeting01_prevalidation \
  --profile results/meeting01_prevalidation/snn-ae/profile.json
.venv/bin/python3 scripts/pipeline/meeting01/00_meeting01_prevalidation_report.py --report \
  --profile src/experiments/meeting01/profiles/meeting01-prevalidation.json \
  --out-root results/meeting01_prevalidation \
  --data-dir ../../documentation/07-articlesProduced/meeting01/data
```

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

`02_` and `04_` overwrite the paper's placeholder files in
`documentation/07-articlesProduced/meeting01/data/` (`\cfgDataStatus` turns from `placeholder`
to `results`), so the next `pdflatex` pass prints the run's numbers. Two rules apply there
(since results_format 6, 2026-11-02): the headline tables leave out the **development fold**
(fold 0 of FSDD and eegmmidb), which `04_` analyses again in `paper_loso_sensitivity.tex`; and
`02_` refuses folds whose recorded configuration differs (a redeploy that changed the profile
mid-run). What each file feeds:
[Meeting01 § The paper's data contract](../Experiments/Meeting01.md#the-papers-data-contract).

> **Results made by a `meeting01` binary older than 2026-10-07 cannot be finished.** Their
> manifests carry no `results_format` (binaries before 2026-10-06) or a 2 (the interim binary
> of commit `b434792e`; the current format is 3), so `03_`, `02_` and `04_` each stop before writing
> anything and name the fold and the reason. There is no repair path. Without a stamp, those
> folds normalized padded windows differently (before commit `6f332734`), left the per-window
> `encoding` empty and wrote `train_ms = 0` for the baselines. With a 2, the LSTM/GRU/Transformer
> frames were a strided gather and the LSTM/GRU cost counted one stack of two. Rebuild
> `meeting01` from the current tree on the cluster (a redeploy does it), then rerun those
> folds: either the whole run without `RESUME` (it clears the GA search checkpoints and reruns
> every fold), or delete only the affected folds' `*_comparative_metrics.csv` and use `RESUME=1` —
> a rerun fold recomputes every run, and an interrupted search of another `results_format` is
> refused (delete its `meeting01_ga_*` files), so no old row survives the rerun. Why:
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
  open question on its own. `INPUT`/`OUTPUT` are **resolved** too (2026-10-13): they are
  hard requirements (job-nanny exits 3 without them), and the sbatch script now sets
  both — see [job-nanny](#job-nanny-what-it-really-does-and-how-this-job-uses-it-2026-10-13).
- Whether reading the datasets from `/home` once per fold counts as the heavy `/home` I/O the
  job-nanny page says may get a job cancelled. Copying them into `INPUT` would mean copying
  ~24GB per job start and pointing the profile's absolute roots elsewhere.
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
