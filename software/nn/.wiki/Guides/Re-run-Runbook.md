# Re-run Runbook

Commands to regenerate every experimental result, in dependency order. Everything here is
**expensive** (hours) — the `expensive-experiment-guard` hook in `.claude/hooks/` blocks an
agent from launching these without explicit confirmation, by design. Run them yourself.
Commands run from `software/nn/` unless a `cd` is shown.

> **Current state (2026-11-13).** The thesis results of record are the runs of 2026-11-12/13
> under the revised protocol ([Experiment05 › Status](../Experiments/Thesis.md#overview)): Phase 00,
> Phase 01 and BIOMEX-DB, committed in `results/thesis/`. The table below is *why the July re-run
> happened*; its counts are July's (the grid has changed since).
>
> This page is the permanent home of the former repo-root `run.md`. See
> [Engineering Fixes Log](./Engineering-Fixes-Log.md) for *why* a re-run was needed at
> all — the short version: the previously stored results were invalidated by fixes
> landed 2026-07-16 (D1/D3/D6 below), not because they were noisy, but because they
> were *wrong*.

| Fix | What it silently did to the old results |
|---|---|
| **D3** — `snn_lr_scale` | Applied the biophysical lr reduction to **every** parameter, so all 24 AE profiles trained every weight at an effective **1e-4 while declaring 1e-3**. Also hit the Meeting01 paper's 3 SNN profiles. |
| **D1** — dead latent | The 18 `snn-ae` profiles now set `firing_rate_reg_lambda=0.5`; the old runs predate the field. |
| **D6** — EEG `scale` axis | 92 EEG bark/mel profiles retired (grid **300 → 208**); their results had no profile left to generate them. |
| **MSELoss/MAELoss** | Silently clipped their own gradient at norm 1.0 — unconditionally, non-configurably, and overriding `grad_clip_norm=0`. `MSELossImpl` is Trainer's default loss, so this touched **every** trained autoencoder. |
| **Adam weight decay** | Applied *after* the gradient step instead of against θ_{t-1} (AdamW's definition; torch agrees). |
| **LSTM activations** | Now exact sigmoid/tanh by default (matching `torch.nn.LSTM`) instead of softsign approximations; the fast path's backward was also the derivative of a function the forward never computed. |
| **Unseeded AE weight init** (2026-07-19) | `extract_features()` seeded the spike frames but never the model weights, so `initializer_seed` stayed `nullopt` and the initializers fell back to `std::random_device`. The **same profile with the same seed produced different features every run.** Affects the **24 Phase 00 autoencoder profiles only** — the 184 handcrafted profiles train nothing and already reproduce bit-for-bit. See [Test Quality and Determinism](./Test-Quality-and-Determinism.md). |

---

## 0. Prerequisites

```bash
cmake --preset=max-performance          # only if out/build/max-performance is missing
```

Sanity-check before burning hours — all of these should already pass:

```bash
ctest --test-dir out/build/max-performance -R "Micro|PyTorchParity|Optim|Trainer|Thesis" -j4
```

The bases must exist where the profiles point: `~/database.sqlite` (Coretto) for Phase 00/01
and `~/biomex.sqlite` for the replication. How to build and verify them:
[Data Bases](./Data-Bases.md).

---

## 0b. Smoke-test the pipeline first (~8 min) — recommended

Every profile has a `smoke/` mirror with the same code-path selectors but tiny parameters
(60-sample cap, 2 epochs, 2 folds). It exercises the whole chain in minutes instead of hours,
so a plumbing bug surfaces before you commit a night to it. Smoke profiles write to
`results/thesis/smoke` — **never** to `results/thesis/phase00`.

```bash
./scripts/testing/run_thesis_smoke.sh phase00
./scripts/testing/run_thesis_smoke.sh phase01
./scripts/testing/run_thesis_smoke.sh biomex
```

You can dry-run the rank + winner steps on the smoke results too, but **redirect the two
destructive arguments** (see the warnings in [Running Experiment05 Profiles › Full pipeline](./Running-Thesis-Profiles.md#full-pipeline)):

```bash
python3 scripts/pipeline/thesis/01_thesis_phase00_rank.py \
  --profiles-dir src/experiments/thesis/profiles/smoke/phase00 \
  --results-dir  results/thesis/smoke \
  --out          /tmp/smoke_winners.json

# Do NOT pass a smoke winners file to 00_thesis_generate_profiles.py: it writes the real tree.
# (GeneratedTree in test_thesis_phase_scripts.py would then fail: the manifest names another file.)

python3 scripts/pipeline/thesis/thesis_build_phase00_paraconsistent_tables.py \
  --profiles-dir src/experiments/thesis/profiles/smoke/phase00 \
  --results-dir  results/thesis/smoke \
  --tables-dir   /tmp/smoke_tables                                  # NOT the thesis tables
```

Every argument of the three scripts is required (no default paths), and each refuses an
incomplete result set: the files each profile declares (its `run_tag`, plus `_rep<k>` for
`repeats > 1`) must all exist.

Smoke *numbers* are meaningless (2 epochs on 60 samples) — only the plumbing is being tested.
Clean up with `rm -rf results/thesis/smoke` afterwards.

> **The tables script reads the profiles, not the results directory** (since 2026-11-11). It
> used to glob `^e05_e05_(p00_.+)_rep(\d+)_summary\.json$`, so smoke runs (tag `smoke_…`,
> `repeats` 1) never matched and it printed a warning and exited 0. It now derives each file
> name from the profile, so the smoke set works — point `--tables-dir` away from the thesis.

---

## 1. Meeting01 — Meeting01 paper (do this FIRST)

Highest priority: it is the only artifact with an external audience, and its current
SNN-vs-LSTM/GRU/Transformer table used to be **unfair to the paper's own contribution** — the
SNN side trained every weight 10× slower than the baselines it is compared against (D3, now
fixed).

> The old `01_meeting01_run_article_profiles.sh` / `02_meeting01_build_lstm_vs_snn_paper_data.py`
> chain (and the `article-*.json` profiles it ran) was **deleted 2026-09-23**: those profiles
> never set `dataset.cv_fold`, so they pooled every window across speakers/recordings, shuffled,
> then split — the same speaker/recording could land in both train and validation. A reviewer
> flagged this as a strong-reject defect on submission 71 (see [Meeting01](../Experiments/Meeting01.md)).
> `meeting01-loso.json` (nested leave-one-group-out) is now the only production profile; there is
> no non-LOSO fallback left in the code.

**Do not pre-build anything for this one.** The script builds its own binary from the preset
named in `MEETING01_BUILD` (default `max-performance`); a binary you built from some other
preset is ignored unless you set that preset.

`train_ms` / `infer_ms` / latency feed the paper's tables directly, so every (dataset, fold) run
must land on the **same** backend. **`max-performance` (CPU/XTensor) is the reference and the
default** — the same backend the thesis uses. The script prints a warning if `MEETING01_BUILD`
is anything else, and that choice must then be reported as a different backend.

```bash
cd software/nn
# Weeks-scale across all datasets x folds (see the profile's own _total_runs_breakdown).
# Refuses to start without EXPERIMENT_CONFIRMED=1. Runs profiles/meeting01-loso.json once
# per (dataset, outer fold), then chains into 03_/02_/04_ (PCA/mean baselines, paper DAT
# files, significance tests) when the grid finishes.
EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
```

Per-epoch progress is logged as `[loso] ... epoch N/M` lines on stderr (survives `nohup`, unlike
a live progress bar). The script is non-interactive throughout — no build-selection prompt —
everything is env-var driven:

```bash
# Reuse the existing binary instead of rebuilding — only when you know it is current:
SKIP_BUILD=1 EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh

# Resume an interrupted run — skips any (dataset, fold) that already finished:
EXPERIMENT_CONFIRMED=1 RESUME=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh

# Stop after training, before the Python post-processing (e.g. on a remote/HPC checkout
# with no .venv/numpy) — see .wiki/Guides/GridUnesp-Deployment.md:
EXPERIMENT_CONFIRMED=1 SKIP_POSTPROCESS=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
```

Every start records which code it ran: the script prints a revision label
(`commit[+dirty]/tree-hash`), exports it into each fold's events file, and appends one line to
`results/meeting01/source_revisions.log` (restarts included). In a **deployed** tree (one with a
`SOURCE_REVISION` file at its root, as `gridunesp_deploy.sh` leaves it) `SKIP_BUILD=1` is
checked, not trusted: the script refuses to start unless the binary's build stamp equals
`SOURCE_REVISION`. In a git checkout there is nothing to compare, so "only when you know it is
current" still means you. Details:
[GridUnesp Deployment](./GridUnesp-Deployment.md#redeploying-what-is-stamped-what-is-refused-when-it-is-safe-2026-10-07).

The first run of a preset also **configures** it (a few minutes on top of the runtime); later
runs are incremental no-ops. By default the script clears `results/meeting01/checkpoints/` and
`results/meeting01/models/` first — pass `RESUME=1` (implies `KEEP_CHECKPOINTS=1`) instead of
`KEEP_CHECKPOINTS=1` directly, since a bare `KEEP_CHECKPOINTS=1` re-run leaves the per-window CSV
incomplete for any fold it re-runs.

> Article-pipeline results predating this deletion (`results/meeting01/article_*_comparative_metrics.csv`)
> are stale artifacts of the removed leakage-prone split — do not cite them; regenerate from
> `meeting01-loso.json` instead.

Then recompile the paper:

```bash
cd ../../documentation/07-articlesProduced/meeting01
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## 2. Thesis — Phase 00, Phase 01, BIOMEX-DB

The commands, their order and their gating are in one place,
[Running Experiment05 Profiles › Full pipeline](./Running-Thesis-Profiles.md#full-pipeline);
this section only says what a complete re-run is:

```bash
cmake --build out/build/max-performance --target thesis -j$(nproc)
./scripts/testing/run_thesis_profiles.sh start-over-all   # phase00 → winners → phase01 → biomex, from scratch
```

Then, by hand: the Phase 01 and BIOMEX tables and significance tests (steps 6 and 8 of that
page), and the thesis numbers and PDF ([Thesis Document › Results to text](./Thesis-Document.md#results-to-text)).
On the cluster each phase is one Slurm array instead
([Experiment05 › Running on GridUnesp](../Experiments/Thesis.md#running-on-gridunesp-2026-11-12)).

Resumable: every completed profile is checkpointed to
`results/thesis/run_profiles_<scope>.state`; on restart the runner offers resume or start-over
(`RESUME=1` forces resume, `FRESH=1` start-over; `start-over-all` sets `FRESH=1` on each leg).

**Significance tests are declared, not explored.** `03_thesis_phase01_significance.sh` and
`04_thesis_biomex_significance.sh` hold the only tests the thesis quotes, each a family with a
reference fixed in advance, Holm within the family. The underlying tool,
`thesis_cross_profile_significance.py`, takes every choice explicitly
(`--metric`, `--reference best|<run_tag>`, `--pairing paired|corrected|independent`, `--select <regex>`);
`corrected` (Nadeau–Bengio) is the one for fold-matched conditions, `independent`
(Mann–Whitney) for conditions whose test sets differ (seen vs unseen phrases). Its docstring
explains each refusal.

---

## What to expect

> **Results must not change independently of a run.** Since the 2026-07-19 re-run, the same
> profile with the same seed produces **byte-identical** output. If you re-run and the numbers
> move, that is a bug, not noise — start at the
> [reproducibility contract](./Test-Quality-and-Determinism.md#reproducibility-contract-results-must-not-change-independently-of-a-run)
> and check whether an initializer has fallen back to `std::random_device` again. The
> `ThesisReproducibility.*` tests exist to catch exactly that before it reaches a run.


**Numbers will move — that is the point.** Every fix above changed what actually executes.
A shift from previously published figures is the correction landing, not a regression.

**The handcrafted numbers should return bit-identical.** Handcrafted extraction trains
nothing, so none of the fixes (all of which touch training) can reach it. If those move,
something is wrong — stop and investigate rather than accept it.

**Watch for a `[TIE]` warning** from `01_thesis_phase00_rank.py`. It means the winner was chosen
by sort tie-break rather than merit — the exact failure that produced a false "Bark won for
EEG" (D6). It should not fire now that the degenerate EEG bark/mel profiles are retired;
if it does, do not attribute the win to any property only the chosen label has.

**Phase 01 metrics live under `results[]`, not at the top level.** A `*_summary.json` has
top-level `run_tag`/`training`/`dataset`/…, and the EER/AUC/accuracy sit inside the
`results` **array** (one entry per feature set): `results[0].mean_eer`, `.mean_auc`,
`.ci95_eer`, `.std_eer`, ….

**Result files are now self-describing.** Each `*_summary.json` carries a `training` block
with `optimizer_type`, the **resolved** `learning_rate`, and a `learning_rate_source`
(`profile` vs `optimizer_default`). The old summaries recorded no training parameters at
all — the provenance gap that let D3 go unnoticed (see [Engineering Fixes Log](./Engineering-Fixes-Log.md)).

**Thesis now records the same run diagnostics as the Meeting01 paper.** Each `*_summary.json`
also carries a `config_hash` (provenance fingerprint), per-run `param_count` and
`mean_train_ms`/`mean_infer_ms`, and per-fold `train_ms`/`infer_ms` + `final_train_loss`.
A new `*_learning_curves.dat` holds per-epoch train/val loss curves plus, for the **DSNN**,
per-epoch `spike_rate` and `sops` (the Meeting01 epoch-history + SNN-efficiency analog); `*_metrics.csv`
gains `train_ms`/`infer_ms` columns, and the summary carries run-level `mean_spike_rate`/`final_sops`.
(Meeting01's in-run SNN-vs-LSTM significance test has no per-run analog here — an Thesis run scores one
feature set — so that comparison is a post-hoc step, §2 above.)

---

## After the re-runs

- Review `git diff src/experiments/thesis/profiles` (the winners written in) and the new
  `results/thesis/` files, then commit results, tables and `thesis_numbers.tex` together, so the
  text never cites a number its results do not hold.
- [Engineering Fixes Log](./Engineering-Fixes-Log.md) — D3 and D5: the optimizer ablation (item
  51) needs a **per-optimizer** lr grid (a single lr is not comparable: reference defaults span
  adam 1e-3 → lion 1e-4).

## Related

- [Running Experiment05 Profiles](./Running-Thesis-Profiles.md) — day-to-day Thesis operation
- [Experiment05](../Experiments/Thesis.md) — design and current results
- [Engineering Fixes Log](./Engineering-Fixes-Log.md) — why each fix above was needed
