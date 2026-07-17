# run.md — re-run runbook (2026-07-16)

Commands to regenerate every experimental result, in dependency order. Everything here is
**expensive** (hours) — the `expensive-experiment-guard` hook in `.claude/hooks/` blocks an
agent from launching these without explicit confirmation, by design. Run them yourself.

**Why a re-run is needed at all:** the previously stored results were invalidated by fixes
landed on 2026-07-16. They are not being re-run because they were noisy — they were *wrong*:

| Fix | What it silently did to the old results |
|---|---|
| **D3** — `snn_lr_scale` | Applied the biophysical lr reduction to **every** parameter, so all 24 AE profiles trained every weight at an effective **1e-4 while declaring 1e-3**. Also hit the E04 paper's 3 SNN profiles. |
| **D1** — dead latent | The 18 `snn-ae` profiles now set `firing_rate_reg_lambda=0.5`; the old runs predate the field. |
| **D6** — EEG `scale` axis | 92 EEG bark/mel profiles retired (grid **300 → 208**); their results had no profile left to generate them. |
| **MSELoss/MAELoss** | Silently clipped their own gradient at norm 1.0 — unconditionally, non-configurably, and overriding `grad_clip_norm=0`. `MSELossImpl` is Trainer's default loss, so this touched **every** trained autoencoder. |
| **Adam weight decay** | Applied *after* the gradient step instead of against θ_{t-1} (AdamW's definition; torch agrees). |
| **LSTM activations** | Now exact sigmoid/tanh by default (matching `torch.nn.LSTM`) instead of softsign approximations; the fast path's backward was also the derivative of a function the forward never computed. |

See `fixme.md` for the full write-ups, and `software/nn/results/phase00/README.md`.

---

## 0. Prerequisites

```bash
cd software/nn
cmake --preset=max-performance          # only if out/build/max-performance is missing
```

Sanity-check before burning hours — all of these should already pass:

```bash
ctest --test-dir out/build/max-performance -R "Micro|PyTorchParity|Optim|Trainer|E05" -j4
```

The dataset must exist at the `dataset.root` in the profiles
(`.../baseDeDatosHablaImaginada/database.sqlite`).

---

## 0b. Smoke-test the pipeline first (~8 min) — recommended

Every profile has a `smoke/` mirror with the same code-path selectors but tiny parameters
(60-sample cap, 2 epochs, 2 folds). It exercises the whole chain in minutes instead of hours,
so a plumbing bug surfaces before you commit a night to it. Smoke profiles write to
`results/smoke` — **never** to `results/phase00`.

```bash
cd software/nn
./scripts/testing/run_e05_smoke.sh phase00     # 208 profiles, ~7 min
./scripts/testing/run_e05_smoke.sh phase01     # 32 profiles,  ~1.5 min
```

Last verified 2026-07-16: **208/208 and 32/32 passed, 0 failed**.

You can dry-run the rank + winner steps on the smoke results too, but **redirect the two
destructive arguments** (see the warnings in §2a):

```bash
python3 scripts/pipeline/e05/01_e05_phase00_rank.py \
  --profiles-dir src/experiments/05/profiles/smoke/phase00 \
  --results-dir  results/smoke \
  --out          /tmp/smoke_winners.json

cp -r src/experiments/05/profiles/phase01 /tmp/phase01_copy    # apply_winner rewrites IN PLACE
python3 scripts/pipeline/e05/02_e05_apply_winner.py \
  --winners /tmp/smoke_winners.json --profiles-dir /tmp/phase01_copy
```

Smoke *numbers* are meaningless (2 epochs on 60 samples) — only the plumbing is being tested.
Clean up with `rm -rf results/smoke` afterwards.

> **The tables script cannot be smoke-tested.** Smoke run tags are prefixed `smoke_` and carry
> no `_repN`, so they never match its `^e05_e05_(p00_.+)_rep(\d+)_summary\.json$` glob — it is
> real-phase00-only by design. It prints `no phase00 summary.json files found` and exits
> cleanly; that is expected, not a failure.

---

## 1. E04 — Guayaquil paper (do this FIRST)

Highest priority: it is the only artifact with an external audience, and its current
SNN-vs-LSTM table is **unfair to the paper's own contribution** — the SNN side trained every
weight 10× slower than the LSTM baseline it is compared against (D3).

**Do not pre-build anything for this one.** The script builds its own binary from the preset
*it* selects; a binary you built from some other preset is ignored unless you pick that preset.

It **asks which build to use**, because that choice is part of the measurement rather than a
convenience: the paper reports **`train_ms` / `infer_ms` / latency**, and
`02_e04_build_lstm_vs_snn_paper_data.py` feeds those straight into its tables, so all four
profiles must run on the **same** backend. **`max-performance` (CPU/XTensor) is the reference
and the default — the same backend the thesis (§2/§3) uses**, so both experiments report from
one setup. Picking anything else prints a warning and must be reported as a different backend.

```bash
cd software/nn

# ~2.5 h (LSTM ~10 min, each SNN ~45 min). Asks which build, then configures + builds it,
# runs all 4 article profiles, converts the NPZ artifacts, and aggregates the paper CSV/DAT.
./scripts/pipeline/e04/01_e04_run_article_profiles.sh
```

While it runs, an **`Overall [i/4] … ETA`** line sits at the top of each profile's TUI —
the same mean-time-per-completed-profile estimate `run_e05_profiles.sh` uses, refined at each
profile boundary. It is optimistic right after the fast LSTM and corrects upward once the
first SNN lands; treat it as a guide, not a promise.

The prompt lists every preset, marks which already have an `experiment04` binary `[built]`,
flags the reference, and defaults to it — so pressing Enter is the reference choice.
Non-interactive runs (pipe/CI) skip the prompt and use the reference.

```bash
# Choose non-interactively (required in a pipe/CI):
E04_BUILD=max-performance ./scripts/pipeline/e04/01_e04_run_article_profiles.sh

# Reuse the existing binary instead of rebuilding — only when you know it is current:
SKIP_BUILD=1 E04_BUILD=max-performance ./scripts/pipeline/e04/01_e04_run_article_profiles.sh
```

The first run of a preset also **configures** it (a few minutes on top of the runtime); later
runs are incremental no-ops.

> Your earlier article results (`results/article_*_comparative_metrics.csv`) predate this
> default and may have been produced on OpenCL — the summaries don't record the backend, so
> it can't be told from disk. For a clean paper, run all four fresh on `max-performance`.

Then recompile the paper:

```bash
cd ../../documentation/07-articlesProduced/conference71070Guaiaquil
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## 2. E05 Phase 00 — 208 profiles

```bash
cd software/nn
cmake --build out/build/max-performance --target experiment05 -j$(nproc)

# HEAVY — run overnight. 208 profiles x experiment.repeats.
./scripts/testing/run_e05_profiles.sh phase00
```

Resumable: every completed profile is checkpointed to `results/run_profiles_phase00.state`.
On restart it offers resume vs. start-over (non-interactive default = resume);
`RESUME=1` forces resume, `FRESH=1` forces start-over.

Background it if you prefer:

```bash
nohup ./scripts/testing/run_e05_profiles.sh phase00 > phase00_run.log 2>&1 &
```

See `.wiki/Guides/Running-Experiment05-Profiles.md` for binary selection (`E05_BUILD`,
`E05_BIN`), live progress bars, and failure triage.

### 2a. Rank → winner → tables

> ⚠️ **Two of these commands are destructive if pointed at the wrong path.**
> - `02_e05_apply_winner.py` **rewrites the phase01 profiles in place**. Pointing it at a
>   winners.json from a smoke/partial run silently overwrites the real profiles' extractor.
> - `e05_build_phase00_paraconsistent_tables.py` **defaults its `--tables-dir` to the thesis
>   tables directory**. Run it with anything other than complete phase00 results and it
>   overwrites the committed tables the thesis compiles from.
>
> Both are fine with the arguments below; just do not run them against `results/smoke`.

```bash
# winners.json — selects on d_penalized (NOT raw d_truth, which a collapsed latent games)
# and now reports exact ties explicitly.
python3 scripts/pipeline/e05/01_e05_phase00_rank.py \
  --profiles-dir src/experiments/05/profiles/phase00 \
  --results-dir  results/phase00 \
  --out          results/phase00/winners.json

# Inject the real winner into the 32 phase01 profiles, replacing the daub4/lfcc placeholder.
python3 scripts/pipeline/e05/02_e05_apply_winner.py \
  --winners      results/phase00/winners.json \
  --profiles-dir src/experiments/05/profiles/phase01

# Regenerate the thesis tables (currently frozen from the pre-deletion data).
python3 scripts/pipeline/e05/e05_build_phase00_paraconsistent_tables.py \
  --results-dir results/phase00 \
  --tables-dir  ../../documentation/00-thesis/monography/tables
```

---

## 3. E05 Phase 01 — the thesis's actual experiment

**Only after step 2a**, or the DSNN trains on the placeholder extractor rather than the
Phase 00 winner. Phase 01 has never produced a single result: full code + profile coverage,
no data.

```bash
./scripts/testing/run_e05_profiles.sh phase01     # 32 DSNN profiles → EER/AUC in results/phase01
```

---

## What to expect

**Numbers will move — that is the point.** Every fix above changed what actually executes.
A shift from the previously published figures is the correction landing, not a regression.

**The handcrafted numbers should return bit-identical.** Handcrafted extraction trains
nothing, so none of the fixes (all of which touch training) can reach it. If those move,
something is wrong — stop and investigate rather than accept it.

**Watch for a `[TIE]` warning** from `01_e05_phase00_rank.py`. It means the winner was chosen
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
all — the provenance gap that let D3 go unnoticed.

---

## After the re-runs

- `fixme.md` — D3 and D5 are blocked on this; item 51's optimizer ablation needs a **per-optimizer**
  lr grid (a single lr is not comparable: reference defaults span adam 1e-3 → lion 1e-4).
- The Phase 00 §09 thesis tables' autoencoder rows are provisional until step 2a is re-run.
- `results/phase00/README.md` explains the empty directory; it can go once results exist.
