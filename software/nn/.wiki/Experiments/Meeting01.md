# Experiment04: SNN vs LSTM Comparative

Experiment04 implements a comparative study between Spiking Neural Networks (SNNs) and LSTM autoencoders on time-series data, with support for the Free Spoken Digit Dataset (FSDD).

## Reviewer-driven revision (`meeting01-loso.json`)

> **The problem this fixes.** The original study pooled every window, shuffled,
> then split — so the same speaker *and the same recording* landed in both train
> and validation. Reported error then partly measured memorization, not
> generalization. A reviewer flagged this as a strong-reject defect.
>
> **The fix.** Nested six-fold **leave-one-group-out** cross-validation. A
> *group* is the speaker (FSDD, AudioMNIST) or the EEG subject (eegmmidb, siena;
> the group directory a recording's `.edf` file lives in — see below). Windows
> are partitioned by group *before any pooling*; per fold: test block = one
> group-block, validation block = the next (rotating), the rest train. The SNN's
> architecture (encoder depth/width, encoding, input-transform, `v_th`, `alpha`)
> is NSGA-II-searched on the **validation block only** ([details below](#nsga-ii-architecture-search-added-2026-09-22-grid-removed-2026-09-22)),
> then the winner is retrained on train ∪ validation (early-stopping on a
> recording-disjoint monitor carved from train) and evaluated **once** on the
> test block. A startup assert aborts the run if any group or `recording_id`
> crosses a split; a `*_split_manifest.json` records the partition.

### Three datasets (multi-dataset answer to "one small database")

> **Dataset swap, 2026-09-22 → 2026-09-23.** The grid originally paired the two
> spoken-digit datasets with `mitbih` (single-lead ECG) as its third, structurally
> different signal — the point being "not just audio". 2026-09-23: `mitbih` was
> replaced by *two* EEG datasets instead of one, per direct request, so the grid
> now has **four** datasets. `mitbih`'s loader code (`Meeting01MitBih.{hpp,cpp}`,
> dataset name `"mitbih"`) was **not deleted** — it still parses and dispatches
> correctly, it is simply no longer listed in `meeting01-loso.json`'s
> `evaluation.datasets`. The heading below still says "Three" because renaming a
> stable anchor breaks every internal link to it; read it as historical, the table
> is current.

| key | signal | group | #groups / K | per-recording window cap | root (default) |
|---|---|---|---|---|---|
| `fsdd` | spoken digits, 8 kHz | speaker | 6 / 6 | none | `.../databases/fsdDataset` |
| `audiomnist` | spoken digits, offline-resampled 48→8 kHz | speaker | 60 / 6 | 2 | `.../databases/audioMNIST_8k` |
| `eegmmidb` | EEG, 64 ch (signal 0 read), 160 Hz, EDF+ | subject (parent dir) | 109 / 6 | 40 | `.../databases/eegmmidb` |
| `siena` | EEG, up to 35 ch (signal 0 read), 512 Hz, EDF | subject (parent dir) | 14 / 6 | 40 | `.../databases/siena` |

**Why a new loader, not the existing `EEGLoader`.** The codebase already has an
EEG loader (`nn::dataLoaders::EEGLoader`), but it reads a MAT-file/sqlite
"imagined speech" format for the `thesis`/`paraconsistentGA` experiments — a
different format, a single dataset, not wired into meeting01. `eegmmidb` and
`siena` are public PhysioNet corpora distributed as **EDF** (European Data
Format, Kemp et al. 1992), a different binary layout entirely. Rather than an
offline edf→WFDB conversion step, `Meeting01Eeg.{hpp,cpp}` reads EDF directly:
fixed 256-byte main header, `ns × 256` bytes of per-signal header fields, then
2-byte little-endian samples — parsed and cross-checked against the format spec's
own worked example before being wired in. It reads signal 0 of every `.edf`
file (assumed to be an EEG channel — verified 2026-09-23 against real
downloaded file headers: signal 0 is labelled `EEG Fp1` for Siena and `Fc5.`
for eegmmidb, both genuine 10-20-system electrode positions, not a status/EKG
channel).

**Why the group is a directory name, not a filename.** Both target corpora
organize files as one subdirectory per subject — `PNNN/PNNN-M.edf` for
Siena, `SNNN/SNNNRMM.edf` for eegmmidb — so `EegWindowDataset` takes each
`.edf` file's *immediate parent directory name* as the leave-one-group-out
group, with no dataset-specific filename parsing needed. This is the same
"group = whatever the LOSO split must never let leak across train/val/test"
idea as `fsdd`/`audiomnist`'s per-speaker grouping and `mitbih`'s per-record
grouping — only *where the loader reads the group from* differs.

- `dataset.sources[]` in the profile gives each dataset its `root` / `window_size` /
  `cv_num_folds` / `sample_rate` / `max_windows_per_recording` / `loso_max_{train,val,test}_windows`;
  unset fields inherit the singular `dataset.*`. `Meeting01Config::Dataset::resolve(name)`
  does the merge.
- **Stratified per-fold window caps** (`loso_max_train_windows` 1200, `loso_max_val_windows`
  300, `loso_max_test_windows` 1500 in `meeting01-loso.json`): after the LOSO split,
  `stratified_window_cap` subsamples each partition round-robin across recordings (ordered
  by `recording_id`), so every recording and speaker keeps representation and per-recording
  counts stay as even as the cap allows. Full pooled FSDD is ~27k train windows/fold —
  intractable at batch-size 1 across the GA's population(10)×(1+generations(8))=90 SNN
  trainings × 5 seeds × 18 (dataset, fold) processes. The caps do **not** touch the leave-one-group-out structure
  or the recording-level statistical unit; they bound per-epoch cost. `cap <= 0` = unlimited.
  Per-epoch progress prints as `[loso] <ctx> epoch N/M train=… val=…` (`Meeting01EpochLogger`,
  to stderr — survives nohup, where the `ProgressManager` bars collapse to one line).
- **Grouped folds:** `assign_speaker_fold` partitions the sorted group ids into `K`
  contiguous blocks (reduces to plain leave-one-speaker-out when `#groups == K`).
- **Loaders:** `fsdd`/`audiomnist` reuse `FsddWindowDataset` (AudioMNIST filenames
  `digit_speaker_index.wav` parse identically; convert to 8 kHz mono first, e.g.
  `sox in.wav -r 8000 -c 1 -b 16 out.wav`). `mitbih` uses `MitBihWindowDataset`
  (`Meeting01MitBih.{hpp,cpp}`) — a minimal WFDB format-212 reader (non-recursive
  `.hea` scan, 12-bit two's-complement decode, physical units via header gain/baseline);
  loader still present, `"mitbih"` no longer in the active profile's `evaluation.datasets`.
  `eegmmidb`/`siena` both use `EegWindowDataset` (`Meeting01Eeg.{hpp,cpp}`) — a minimal
  EDF reader (recursive `.edf` scan, header self-check against its own declared byte size,
  reads data records until a short read rather than trusting `n_data_records`, digital→
  physical via the header's min/max, group = parent directory name).
- Tests: `loaders_gtest` (real-loader checks, skipped when a root is absent — plus
  `EegSyntheticEdfDecodesAndGroupsBySubject`, a hand-built-`.edf`-fixture test that runs
  even without either real EEG corpus downloaded), `meeting01_split_audit_gtest`,
  `profile_audit_gtest` (`DatasetSourceResolution…`).

### Running it

One process per **(dataset, fold)**:

```bash
cd software/nn
EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
# loops --dataset {fsdd,audiomnist,mitbih} --cv-fold 0..5, then 03_ (PCA/mean)
# → 02_ (paper tables) → 04_ (recording-level significance).
```

Outputs are tagged `meeting01_loso_<dataset>_fold<f>_*`. Weeks-scale even with the
window caps; run once, checkpoints cleared first.

### Live monitoring

Each `meeting01` process appends structured events to
`results/meeting01/<run_tag>_<dataset>_fold<f>_events.jsonl` (schema v1, written by
`Meeting01Events.cpp` + the `Meeting01EventCallback` training hook). Events:
`session_begin` (search space, seed, caps, git commit, backend), `fold_begin` /
`fold_end`, `config_begin` / `epoch` / `train_end` / `config_end` per trained model,
`epoch_progress` (throttled intra-epoch heartbeat — one line per ~5 s of a slow
epoch: batch fraction, running batch loss, epoch ETA; a fast epoch emits none),
`config_selected` (GA-selected SNN winner), `session_end` / `session_error`. Raw values at
full precision; NaN → `null`. Emitting never gates training — pure side output.

```bash
# attach anytime, in a separate terminal (18 processes write 18 files; the monitor
# tails all of them and folds a single session view)
python3 scripts/pipeline/meeting01/monitor.py --run-tag meeting01_loso
python3 scripts/pipeline/meeting01/monitor.py --plain          # non-interactive / piped
python3 scripts/pipeline/meeting01/monitor.py --self-test      # CI known-answer check
```

The dashboard (needs `rich`, in `scripts/requirements.txt`) has four panels:
**SESSION** (identity, done/running/failed, rough ETA, the grid summary), **TRAINING
NOW** (the active config: epoch bar, plus — for epochs slower than ~5 s — a live
batch bar with running loss and epoch ETA; train/val, best-val @ epoch, train/val
gap, epochs-since-improvement, train/val sparklines), **COMPLETED** (every finished config
ranked by held-out test loss — else best inner-validation loss — plus, once SNN
configs finish, **descriptive** marginal best-val per sweep dimension and
per-(model, encoding) `mean ± std`), and **RECENT** (event tail). Read-only — start,
kill, re-attach freely; `Ctrl-C` exits. `monitor.py --rank N` prints one completed
config's full detail (all metrics + reproducibility). `--no-tui` on the `meeting01`
binary (and any non-TTY stdout) disables its own `ProgressManager` bars so redirected
logs stay free of cursor-control sequences.

### Statistics & paper data

- **Primary estimand:** recording-level paired difference `d_r` (bootstrap over
  recordings, Wilcoxon, Holm across references), **per dataset**. Group-level and
  seed-level are robustness only. `04_meeting01_significance_tests.py` →
  `meeting01_loso_<ds>_significance_recording.tex` + `meeting01_loso_significance.json`.
- `02_meeting01_build_loso_paper_data.py` → `paper_loso_<ds>_{summary,recon_by_encoding,mse_plot}.csv`
  + `paper_loso_<ds>_snn_selection.tex` (mean ± std over 5 seeds; best cell bolded).
- Model inventory: **four trained families** (SNN-AE, LSTM-AE, GRU-AE, Transformer-AE)
  + PCA and mean-frame references. SNN `dense/conv1d/recurrent` are *input transforms*
  selected per fold, not families.
- Paper: `documentation/07-articlesProduced/meeting01/paper.tex`
  (`\resultsForDataset` macro, one block per dataset).

> **Known caveat: AudioMNIST window degeneracy (found 2026-09-16, live LOSO run).**
>
> **The symptom.** A handful of `meeting01_loso_audiomnist_fold*_comparative_metrics.csv`
> rows report `mse == 0.000000` for `model=snn-ae, encoding=direct, architecture=recurrent`
> (mostly `alpha=0.8`) — on the held-out **test** speaker too, across all 5 seeds. That
> looks like the model reached perfect reconstruction; it has not.
>
> **The mechanism.** `load_grouped_windows` (`Meeting01Dataset.cpp`) filters strictly by
> `source_window_index`, in sequential order, with no silence trim. The stratified
> train/val/test split then round-robins **one window per recording** to fill its quota
> (`loso_max_{train,val,test}_windows`). AudioMNIST has far more distinct recordings per
> fold than the quota needs, so index 1+ is never reached — every sampled AudioMNIST
> window, train and test, is window 0 (the first 32 ms at 8 kHz) of its recording.
> Confirmed directly: `source_window_index` distribution over a real fold was `{0: 1200}`
> (train) and `{0: 1500}` (test) — no other index appeared.
>
> **Why AudioMNIST and not FSDD.** Raw WAV inspection: AudioMNIST's first 32 ms is a
> near-silent lead-in (`audioMNIST_8k` recordings begin quiet — raw int16 values in
> roughly ±2..±12) in essentially every file, whereas FSDD's window 0 already carries
> real speech amplitude (hundreds) or a recording-specific pattern. `zscore_inplace`
> forces `std == 1` regardless of original scale, so AudioMNIST's near-silent window 0
> gets amplified into a shape that is virtually identical across the whole corpus. Direct
> measurement on a real fold, `encode_sample("direct")` + `apply_snn_architecture_transform`
> (`"recurrent"`, `alpha=0.8`, `v_th=1.0`) over every train+test window:
>
> | dataset | unique spike patterns | mean pairwise Hamming distance |
> |---|---|---|
> | AudioMNIST | 1 / 2700 | 0.0000 |
> | FSDD | 1144 / 1500 (test only) | 0.21 |
>
> The autoencoder is not learning digit reconstruction for this config — it is
> memorizing one near-constant "silence" spike train, which trivially reconstructs on
> train and on the LOSO-held-out test speaker because that speaker's window is the same
> degenerate pattern. `alpha=0.8` (fastest leak) concentrates the exact-`0.000000` hits
> because it is most sensitive to reproducing that one fixed shape precisely; `0.9`/`0.99`
> blur it just enough to land above the CSV's 6-decimal rounding floor instead of on it.
>
> **Scope.** Not limited to the rows that print exactly `0.000000`. Every model/config
> trained on AudioMNIST in this run is trained on 32 ms of near-silent recording lead-in,
> never the spoken digit — the other combinations likely aren't reconstructing real
> content either, they just don't degenerate cleanly enough to round to zero. Treat the
> whole AudioMNIST reconstruction arm as suspect until this is fixed, not only the exact
> zero rows. `02_meeting01_build_loso_paper_data.py::check_degenerate_reconstruction` now
> flags this automatically (`paper_loso_<ds>_CAVEATS.txt`, >2% of a dataset's test rows
> below `mse=1e-4`) whenever the paper tables are built, so it cannot be missed silently.
>
> **Not yet fixed** (deliberately, to avoid re-running a multi-day live experiment on a
> guess): candidate fixes are (a) silence-trim/VAD before windowing so window 0 lands on
> real content, or (b) round-robin over `(recording, window_index)` pairs instead of
> always draining index 0 first across the whole corpus.

---

## Encoding and Surrogate-Gradient Fixes (found + fixed 2026-09-21, pre-GridUnesp audit)

> **The problem this fixes.** A state-of-the-art audit ahead of the GridUnesp submission
> found `Meeting01Encoding.cpp`'s `poisson` and `latency` encodings — the core independent
> variable of the encoding comparison — diverging from both the literature and this
> project's own paper text (`documentation/07-articlesProduced/meeting01/paper.tex` already
> described the intended full-range design; the code had drifted from it).
>
> **Poisson encoding normalized by `max_v` only** (`clamp(sample/max_v, 0, 1)`). Every
> sub-mean, post-z-score sample — roughly half the signal, by construction — is negative,
> so it was silently clamped to firing probability 0. `latency`, by contrast, already used
> the full `(sample - min_v) / range`. This asymmetry meant `poisson` and `latency` were not
> being compared on the same information content: one discarded half the window's dynamic
> range, the other didn't. **Fix:** `poisson` now uses the same full-range
> `(sample - min_v) / range` as `latency`.
>
> **Latency encoding was a threshold-crossing/step code, not TTFS.** `encoded.at(t,d) =
> (t >= t_spike) ? 1 : 0` fires at `t_spike` and *stays on* afterward (~50% duty-cycle
> spike trains), whereas canonical time-to-first-spike coding (Thorpe; Mostafa 2017; Comşa
> et al. 2020/2021) is exactly one spike per channel, silent otherwise. **Fix:**
> `(t == t_spike) ? 1 : 0` — single spike, matching the paper's own description
> ("A binary step fires once per element") which the old code never actually implemented.
>
> **Surrogate gradient.** `meeting01`'s SNN-AE previously fell back on the framework's
> `ExponentialSurrogate` default (see [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md#surrogate-gradient-methods)).
> ArcTan has been the default spike-derivative estimator in snnTorch since 2023, with
> heavier gradient tails that avoid the saturation Exponential/Boxcar show away from
> threshold. A new opt-in field, `AutoencoderConfig::surrogate_gradient` (default
> `nullptr` — every other experiment's behavior is unchanged), lets `meeting01::make_snn_cfg`
> (`Meeting01Training.cpp`) select `ArcTanSurrogate` without touching `thesis`/
> `autoencoderRunner`/demos. `LifBPTT::compute_grad_step` already calls
> `surrogate_gradient->calculate_scalar()` polymorphically — no per-type special-casing —
> so this is a config-level swap, not a new code path per surrogate.
>
> **Parameter matching.** The paper's Limitations section ("Dimension-matched, not
> parameter-matched") now explicitly covers the whole comparison set (SNN-AE, LSTM-AE,
> GRU-AE, Transformer-AE share $H=64$/$d=32$, not trainable-parameter count $P$), not just
> the Transformer-AE as before — `parameter_count_gtest.cpp` already verifies GRU-AE has
> fewer parameters than LSTM-AE at matched dims (3 gates vs. 4), and the SNN-AE carries its
> own biophysical parameters ($R$, $C$, $V_{th}$) with no non-spiking analogue.
>
> **Verified.** New direct coverage for `Meeting01Encoding.cpp` (`meeting01_encoding_gtest`,
> previously zero) plus new `ArcTanSurrogate` unit + `LifBPTT` integration tests
> (`layers_spiking_basic_gtest`, `spiking_mechanisms_gtest`) — 111 tests, zero regressions.
> Re-validated end to end with the GridUnesp docker toolchain sim (138/138 build steps).
>
> **Consequence for the Results tables below.** They were generated under the *pre-fix*
> encoding implementation — the Poisson numbers in particular reflect the max-only bug, and
> the specific MSE/R²/spike-rate values will change on the next LOSO run. Treat them as
> historical (pre-2026-09-21), not current, until the pipeline is rerun.

---

## The Missing Time Axis (found + fixed 2026-09-22, second pre-GridUnesp audit)

### Start with the thing that was wrong

A spiking neuron does not compute an answer from one snapshot. It charges up, fires,
resets, charges again. Take that sequence away and it is no longer a spiking neuron — it
is a step function with extra parameters that never get used.

`meeting01` ran with `time_steps = 1`.

That single number is the root of everything in this section. The framework itself warns
about it ([Time-Steps](../Concepts/Time-Steps.md): *"Default 0 = unset and RAISES — never
assume 1"*), and `meeting01` was assuming 1 anyway.

### Where the confusion came from

A window is 256 audio samples. The old code treated *those 256 samples* as the time axis,
and told the network there was 1 time step. Two different things were both called "time":

```
OLD  encode_sample(window)  ->  (256, 1)      network time_steps = 1
     row = window sample #             "1 neuron, observed for 256 steps"
     col = the single channel

NEW  encode_sample(window, T=16) -> (16, 256)  network time_steps = 16
     row = simulation step t                   "256 neurons, simulated for 16 steps"
     col = window sample #  (= one input neuron)
```

Canonical time-to-first-spike coding says: **each input neuron fires exactly once.** Under
the old layout there was exactly *one* input neuron, so a whole 256-sample window produced
**one spike**. Measured on a real z-scored speech window: **1.07 spikes per window, 99.58%
zeros, and frequently 0 spikes.** The SNN was being handed a nearly empty tensor and asked
to reconstruct a signal from it.

The test suite did not catch this because the one latency test built its input as
`sample.at(t, d) = values[d]` — each channel held *constant across time*. That is the single
input distribution under which the broken implementation looks correct.

### Same tensor, two meanings

| | Old (`T = 1`) | New (`T = 16`) |
|---|---|---|
| What a row is | one window sample | one simulation step |
| What a column is | the mono channel | one input neuron (one window sample) |
| Input neurons | 1 | 256 |
| Latency spikes per window | **1** (measured 1.07) | **256** (one per neuron, asserted) |
| Poisson draws per feature | 1 Bernoulli coin flip | 16 draws → a real firing *rate* |
| `v = v*beta + vin` | `v` is always 0 at entry, so `beta` multiplies zero | `v` carries 15 steps of history |
| `alpha` (→ `beta`) | **inert** | governs the leak |
| `voltage_threshold` | **never assigned at all** | drives the spike/reset |

### The four silent failures

Every one of these produced a completed run with a plausible MSE. None of them crashed,
warned, or logged anything. That is what makes them expensive: a GridUnesp job would have
burned weeks and published numbers nobody could trace back to a mechanism.

| # | Failure | Evidence | Loud or silent |
|---|---|---|---|
| B1 | Latency encoding was a coincidence detector, not TTFS | 1.07 spikes / 256-sample window; 0 spikes on speech-like input | **Silent** |
| B2 | Models were trained to reconstruct their **own spike code**, not the signal | see below | **Silent**, and it biased the GA |
| B3 | LIF membrane leaked across independent samples for a whole epoch | `has_reset_state` trait existed but `Trainer` never called `reset_state()`; `LifBPTT` only re-zeroes `v_mem` when the tensor *shape* changes, and the shape is constant all epoch | **Silent** |
| B4 | `voltage_threshold` was never written into the model config | zero occurrences of the assignment in `Meeting01Training.cpp` | **Silent** |
| B5 | `time_steps = 1` contradicted the framework's own invariant | `.wiki/Concepts/Time-Steps.md`, CLAUDE.md | **Silent** |

B3 had a second edge: the evaluation paths *did* reset explicitly, so training and
evaluation were running the same network under two different regimes.

### B2 in numbers: the 261× free win

`evaluate_snn` scored `mse_between(encoded, reconstruction)` — the target was the *encoded*
tensor. Different encodings have wildly different variance, so a model that learns nothing
at all still scores very differently depending on which encoding it drew:

| Encoding | Target | MSE of a trivial (mean-predicting) model |
|---|---|---|
| `direct` | z-scored analog values | **1.000** |
| `poisson` | Bernoulli 0/1, p ≈ 0.5 | **0.247** |
| `latency` | 0/1 with ~99.6% zeros | **0.0038** |

The GA minimizes validation MSE. Selecting `latency` therefore bought a **261× lower score
for free**, with no reconstruction skill involved. The search would have reported "the GA
discovered latency coding is best" when it had discovered that sparse targets have small
variance.

**Fix (user decision, 2026-09-22): reconstruct the ORIGINAL signal.** Every model — the
SNN *and* all three baselines — is now scored against `make_reconstruction_target(window, T)`:
the original z-scored window, repeated across the T steps. The target no longer depends on
the encoding at all, which is exactly the property `TargetVarianceIsIdenticalAcrossEncodings`
asserts. `val_mse` is now comparable across encodings, and the GA's encoding gene competes
on reconstruction quality alone.

### What changed, file by file

| File | Change |
|---|---|
| `src/core/training/Trainer.hpp` | **Core framework.** New `reset_model_state()` (calls `reset_state()` when the model has it) invoked before all four `forward()` sites; new `stack_time_major(parts)` building `(T*B, F)` with `out.at(t*B + b, f)`, throwing on shape disagreement and degenerating to the old `(B, F)` stacking at `T == 1`. Fixes B3 for every experiment, not just this one. |
| `Meeting01Encoding.cpp` | `encode_sample(sample, encoding, seed, time_steps)` → `(T, F)` time-major; throws below `T = 2`. `latency` = exactly one spike per feature at `t_f = round((1 − scaled_f)·(T − 1))`; `poisson` = T Bernoulli draws per feature; `direct` holds the analog value at every step. New `make_reconstruction_target` and `reduce_time_major_output`. `conv1d_temporal_smooth` now smooths along the signal axis (columns) after the relayout. |
| `Meeting01Training.cpp` | `make_snn_cfg` finally assigns `voltage_threshold` (B4) and sets `time_steps` from config, `delta_t = 1`, `R = 1`, `C = −1/ln(alpha)` so `beta = exp(−Δt/RC) = alpha` exactly. SNN training moved from `fit_autoencoder` to `fit_supervised` with explicit `(encoded, target)` pairs. |
| `Meeting01AeCommon.hpp`, `Meeting01Evaluation.cpp` | `evaluate_ae` / `per_window_errors_ae` / `evaluate_lstm` / `evaluate_snn` / `per_window_errors_snn` all take `time_steps` and score against the original-signal target. Baselines keep the `to_lstm_frames` framing of the `(T, F)` tensor, so LSTM/GRU/Transformer parameter counts and the H=64 / latent=32 matching with the SNN are unchanged. |
| `Meeting01GaSearch.cpp` | `tournament()` throws instead of spinning forever when the exclusion is unsatisfiable (H1). Winner's-curse mitigation (H5): each final Pareto-front member is re-scored on `winner_seeds` seeds and its `val_mse` replaced by the mean. |
| `Meeting01Config.cpp` | Rejects `time_steps < 2`, `winner_seeds < 1`, and `population_size < 2` combined with `generations ≥ 1` — the last used to validate cleanly and then hang forever in `tournament()`. |
| `Meeting01Metrics.cpp` | New `estimate_snn_macs(input_features, encoder_widths, time_steps)` summing real per-layer projections × steps. The old `(hidden_size, layers)` proxy gave `{128, 8}` and `{128, 120}` the same cost, so the GA's second objective could not tell a cheap architecture from an expensive one (H3). |
| `Meeting01Experiment.cpp` | `SnnSelection` now records `encoding` (H6), and the model-selection manifest's `selected` block records `encoding`, `encoder_widths` and `time_steps` alongside `architecture`/`v_th`/`alpha`. Without those three the published network could not be rebuilt from the manifest — the GA searches a free-form shape *and* its encoding, so "architecture: dense" alone says almost nothing. Note the manifest's top-level `encoding` is the per-run loop label, not the winning genome's gene; they can differ. |
| `profiles/*.json` + `profile_audit_gtest.cpp` | Four shipped profiles (`debug_nested`, `lstm-lightweight`, `minimal-dat-test`, `test-dat-writers`) plus flat-schema `debug.json` failed `validate()` outright (`early_stop_patience >= epochs`) and nothing caught it — the audit only covered the five *article* profiles by name. New `ProfileDirectoryAudit` walks the directory instead of a list, dispatching flat vs nested exactly as `Meeting01Cli::load_config` does, and separately requires every SNN-bearing profile to declare `time_steps >= 2` explicitly. `debug_nested.json` also gained `loso_max_*` caps: without them, `--cv-fold 0` silently ignores the pooled sample caps and trains on every window of FSDD (~105k batches/epoch) on a profile named "debug". |
| `src/bindings/bind_meeting01.cpp` | `encode_sample` now requires `time_steps` (deliberately positional *before* `seed`, so an old caller breaks loudly rather than silently getting a different shape); `snn_ae_forward` feeds the `(T, F)` tensor in as is and returns `reconstruction_window` / `target_window` / `time_steps`. |

### Paying for the 16× (the budget compensation)

Real temporal simulation multiplies every forward and backward pass by T. At T = 16 that is
a 16× bill, on a job that already ran for weeks. Two levers were used to pay it — the
profile's window caps and the GA budget:

| Knob | Before | After | Why |
|---|---|---|---|
| `model.time_steps` | 1 (hardcoded) | **16** | the whole point |
| `loso_max_train_windows` | 1200 | **200** | the dominant cost (30 epochs × fwd+bwd × T) |
| `loso_max_val_windows` | 300 | **150** | evaluated every epoch for early stopping |
| `loso_max_test_windows` | 1500 | **1500 (unchanged)** | forward-only, and these windows *are* the recording-level statistical unit — cutting them buys little compute and costs statistical power directly |
| `ga.population_size` × `(1 + generations)` | 10 × 9 = 90 | **8 × 6 = 48** | fewer trainings per cell |
| `ga.winner_seeds` | — | **3** | new cost: `\|front\|` × 2 extra trainings per cell, buying the winner's-curse fix |

Net effect ≈ 1.8× the old wall-clock, i.e. still weeks, not months. The window caps apply
to the SNN and the baselines alike, so the reduction is matched across the comparison — all
arms train on the same 200 windows. Absolute reconstruction quality will be lower than the
old tables for every model; the *comparison* stays fair.

### New config keys

| Key | Default | Meaning |
|---|---|---|
| `model.time_steps` | 16 | simulation steps per window. **< 2 is rejected at validation**, not clamped |
| `evaluation.ga.winner_seeds` | 3 | seeds each final Pareto-front member is re-scored on before `pick_winner`. 1 disables the mitigation |

Only 4 profiles remain after the 2026-09-23 unused-profile cleanup (the other 6
non-production ones had no real consumer — nothing loaded them by name, they were only
swept by the directory-wide audit). `meeting01-loso.json` (production) and
`lstm-bench.json` (the cited benchmark behind [LSTM Performance](../Guides/LSTM-Performance.md))
both declare `time_steps: 16` to match production. `lstm-compare.json` (the CLI's own
default profile) declares `time_steps: 4` with `winner_seeds: 1` to stay fast. `debug.json`
is flat-schema (see below) and has no `model.time_steps` field at all — it inherits the
struct default (16) since the flat-schema branch is exempt from the explicit-declaration
requirement.

### `snn_time_steps` → `time_steps`: the name was lying about who it affects

The field used to be called `model.snn_time_steps`. The `snn_` prefix said "this is an SNN
knob, the baselines are not affected". That was false, and the false half is the dangerous
half: `run_baseline` (`Meeting01Experiment.cpp`) reads the very same field, because the
baselines are trained on the same encoded tensor. So it sets the **LSTM/GRU/Transformer
unroll length** as well:

```
sequence length fed to the baseline = T * window_size / lstm_frame_size

T = 1,  window 256, frame 8  ->   32 frames    (the pre-2026-09-22 layout)
T = 16, window 256, frame 8  ->  512 frames    (now)
```

Two profiles were inheriting it invisibly and were caught only by the new audit:

- `article-lstm-ae.json` — a **production** profile feeding the paper's LSTM-AE numbers.
  It happened to inherit 16, matching `article-snn-*`, so the comparison was never wrong;
  but nothing in the profile said so, and a change to the struct default would have
  silently desynchronised the two arms of the published comparison.
- `lstm-bench.json` — the LSTM throughput benchmark behind
  [LSTM Performance](../Guides/LSTM-Performance.md). Its recorded wall time was measured
  at a 16× shorter unroll and is now stale. Flagged there.

`EveryProfileDeclaresTimeStepsExplicitly` now requires the field on every nested profile,
LSTM-only ones included.

**The field was renamed to `model.time_steps` on the user's decision (2026-09-22).** The
Python demo `src/demos/pyDemos/multimodal_eeg_audio/` keeps its own unrelated
`snn_time_steps` and was deliberately left alone.

A profile still carrying the old key is **rejected**, not quietly ignored:

```
Meeting01Config: 'snn_time_steps' was renamed to 'time_steps' on 2026-09-22 (the field
sets the unroll length for the LSTM/GRU/Transformer baselines too, not only the SNN's
membrane depth). Remedy: rename the key to 'time_steps' in this profile. Do NOT delete
it — dropping the key would silently fall back to the default of 16 steps.
```

Ignoring the stale key would have meant falling back to the 16-step default — a run that
completes and reports a plausible number under a temporal resolution nobody chose, which
is precisely the failure class this audit existed to remove. Copies of these profiles live
on the cluster, so the old key will resurface; `RenamedKey.OldSnnTimeStepsKeyIsRejectedNotIgnored`
covers both the stale-key-only and both-keys-present cases.

### Dev-profile GA budgets were inherited, and the CLI default was the worst case

Four profiles declared an SNN arm but no `evaluation.ga` block, so they silently ran the
struct default of population 10 × (1 + 8 generations) = **90 evaluations**. The worst of
them is `lstm-compare.json`, which is what the CLI runs when given no
`--comparative-config` at all (`Meeting01Cli.cpp`: `kDefaultComparativeProfileStem`).

Measured on this machine (FSDD, window 256, 500 train windows, `T = 4`, 12 threads):
**SNN ≈ 4 s/epoch, LSTM-AE ≈ 62 s/epoch.** Extrapolating over each profile's own
`epochs`/`repeats`/`encodings`:

| Profile | GA budget before → after | Est. SNN time | Est. baseline time | Est. total after |
|---|---|---|---|---|
| `lstm-compare` (CLI default) | 90 → **24** (6×(1+3)) | 30 h → 8 h | 15.5 h | **≈ 24 h** |
| `lstm-default` | 90 → **8** (4×(1+1)) | ~20 h → 1.8 h | ~3.4 h | **≈ 5 h** |
| `lstm-deep` | 90 → **8** (4×(1+1)) | ~20 h → 1.8 h | ~3.4 h | **≈ 5 h** |
| `lstm-lightweight` | 90 → **2** (2×(1+0)) | minutes → seconds | seconds | **seconds** |

`SnnProfilesDeclareTheirGaBudgetExplicitly` now requires the block, so a new profile
cannot inherit a 90-evaluation search by omission.

**The remaining ~24 h in `lstm-compare` is deliberate (user decision, 2026-09-22).** After
the GA fix, what is left is the profile's *own* `epochs: 100` × `repeats: 3` × 3 encodings
on the baseline loop — 15.5 h that no GA budget can touch. Cutting it to ~2.5 h (30 epochs,
1 repeat) was offered and **declined**: those numbers define what the profile measures, and
1 repeat would drop the standard deviation. So the CLI default is a ~24 h run by choice,
not by omission — the difference that matters, since the same 24 h reached by inheriting a
struct default is a trap and this one is a decision.

### Why `winner_seeds` exists

The GA scores each genome on one seed, then reports the best of ~48 noisy scores. The best
of many noisy draws is biased high by construction — the *winner's curse*. Re-scoring only
the final front (not all 48 genomes) on 3 seeds and publishing the mean makes the selected
architecture reflect expected quality rather than one lucky initialization, at a cost
proportional to the front size rather than the population.

### Verified

`meeting01_encoding_gtest` was rewritten from scratch (the old tests are what let B1
through): time-major shape contract, `T < 2` rejection, exactly-one-spike-per-feature on a
*varying* signal with an explicit `EXPECT_EQ(count_spikes(e), 256)` regression guard,
Poisson firing rate tracking normalized amplitude over 4000 steps, and the
encoding-independence of the target. `meeting01_ga_gtest` gained the first direct coverage
of the selection machinery (`pick_winner` ordering and its empty-front throw, non-dominated
sort, crowding boundary preservation, the `population_size = 1` validation guard, the MAC
proxy fix).

Whole tree: **3204 tests pass, 0 failing binaries**, including `trainer_gtest`, `core_gtest`
and `paraconsistent_ga_gtest` — the `Trainer.hpp` change touches every experiment, so those
were re-run deliberately. The `nn_microscope` Python bindings were configured and built
under the `python-bindings` preset to confirm the new `encode_sample` signature compiles.

End to end, the `meeting01` binary was run on a capped nested-LOSO smoke config (FSDD,
fold 0, `T = 4`, 20/10/20 windows, population 1 × 0 generations): split → leakage gate →
analytic-baseline dump → LSTM-AE baseline → GA → retrained winner → metrics CSVs, all the
way to a `model_selection_manifest.json` carrying the winner's `encoder_widths`. **No
production profile was run** — `meeting01-loso.json` remains a multi-week cluster job.

### Consequence for the Results tables below

They are now **twice** superseded: once by the 2026-09-21 encoding fixes, and again by
everything in this section. Under `T = 1` the reported SNN numbers describe a network with
no membrane dynamics, an inert `alpha`, an unset `voltage_threshold`, a latency encoding
emitting ~1 spike per window, and an encoding-dependent target. They are not a weaker
version of the current result — they measure a different object. Treat them as historical
until the LOSO pipeline is rerun.

---


> **This section through [Results](#results) documents the pre-LOSO design** (the
> `article-*.json` profiles, the pooled/shuffled split, and the single-family GA). That
> pipeline was **deleted 2026-09-23**: the profiles never set `dataset.cv_fold`, so they
> pooled every window across speakers/recordings, shuffled, then split — the same
> speaker/recording could land in both train and validation, the leakage defect a
> reviewer flagged as strong-reject on submission 71. `dataset.cv_fold` is now a hard
> requirement everywhere; there is no non-LOSO fallback left in the code, and the
> `article-*.json` profiles no longer exist on disk. The general theory (seq2seq, surrogate
> gradients, layer specs) below is still accurate; the concrete config schema, data-loading
> description, profile table, run commands, and result numbers are **not** — for those, see
> [Reviewer-driven revision](#reviewer-driven-revision-meeting01-losojson) above and
> [Multi-family architecture search](#multi-family-architecture-search-added-2026-09-22-same-day-later-scope-change)
> below, plus [Re-run Runbook](../Guides/Re-run-Runbook.md) for how to actually run it today.

## Theoretical Background

### Sequence-to-Sequence Learning

LSTM autoencoders compress variable-length sequences into fixed-size latent vectors: 

1. **Encoder LSTM**: Processes input sequence, produces final hidden state
2. **Latent Space**: Fixed-dimensional representation of entire sequence
3. **Decoder LSTM**: Reconstructs sequence from latent state

### SNN Surrogate Gradients

Spiking Neural Networks use surrogate gradient methods to approximate the non-differentiable spike function [1]:

$$\frac{\partial S}{\partial V} \approx \frac{\partial \sigma}{\partial V}$$

where $\sigma$ is a smooth approximation (e.g., fast sigmoid).

### Comparative Framework

The experiment compares:
- **LSTM Autoencoder**: Standard recurrent autoencoder with BPTT
- **SNN Autoencoder**: Layer-based spiking autoencoder with Leaky Integrate-and-Fire neurons

#### Latent Space and Compression

The latent space dimensionality is now defined explicitly within the `encoder_layer_spec` and `decoder_layer_spec`. For example, specifying the last encoder layer as `linear:32:identity` explicitly sets the latent size to 32.

Forcing a small latent dimensionality prevents the model from simply copying the input to the output, requiring it to learn the most critical features of the data. In this comparative study, both models are assigned the same latent dimensionality to ensure a fair comparison of their compression efficiency and reconstruction accuracy.

## Implementation

### Comparative Configuration

Config is loaded from a JSON profile. Top-level sections:

```jsonc
{
  "experiment": { "run_tag", "seed", "repeats", "seed_deterministic", "check_determinism" },
  "dataset":    { "dataset_root", "results_dir", "window_size",
                  "max_loaded_train_samples", "max_validation_samples",
                  "latex_data_dir", "save_models" },
  "training":   { "epochs", "early_stop_patience", "learning_rate",
                  "samples_per_batch", "batches_per_epoch",
                  "beta1", "beta2", "epsilon", "max_reconstruct_mean_deviation" },
  "model":      { "loss_function", "latent_dim", "lstm_hidden_size",
                  "lstm_frame_size",
                  "encoder_layer_spec", "decoder_layer_spec" },
  "evaluation": { "datasets", "encodings", "baselines", "snn_architectures",
                  "ga": { "population_size", "generations",
                          "min_layers", "max_layers", "min_width", "max_width",
                          "voltage_threshold_min", "voltage_threshold_max",
                          "alpha_min", "alpha_max",
                          "crossover_prob", "mutation_prob", "tournament_k" } }
}
```

`evaluation.ga` is the NSGA-II search's bounds and budget — see
[NSGA-II Architecture Search](#nsga-ii-architecture-search-added-2026-09-22-grid-removed-2026-09-22)
below. It is only consulted when `snn_architectures` is non-empty; omitting it falls
back to `Meeting01Config::Ga`'s struct defaults (population=10, generations=8), but
every SNN-bearing profile shipped in `profiles/` declares it explicitly.

Only listed keys are parsed. All other JSON keys (including `_`-prefixed doc strings) are silently ignored.

Parsed by: `src/experiments/meeting01/lib/include/Meeting01Config.hpp` (`from_nested_json`).

#### `model.lstm_frame_size` (default 8)

Samples fed to the LSTM per timestep. The sequence length becomes
`window_size / lstm_frame_size`, so with the article profiles' `window_size=256`
the default gives $T = 32$, $D = 8$.

Before 2026-07-18 this was hard-coded to `input_size = 1`, i.e. the window was
consumed one scalar per timestep ($T = 256$, $D = 1$). Because the dominant cost
per step is the recurrent term $h \cdot U^\top$ — independent of $D$ — that made
the LSTM roughly 7× more expensive than necessary:

| | frame=1 | frame=8 |
|---|---|---|
| sequential steps | 256 | 32 |
| MACs | 8 523 840 | 1 184 256 |
| CPU LSTM train (6 samples, 2 epochs) | 3 711 ms | 478 ms |

Constraints and caveats:

- Must divide `window_size`, enforced by `Meeting01Config::validate()`.
- Encoding is applied to the flat `(window_size, 1)` window **first**, then
  framing — the `direct`/`poisson`/`latency` transforms expect the flat layout.
- Evaluation compares reconstruction in framed space. MSE/MAE/$R^2$ are
  elementwise, so framing both sides leaves them unchanged.
- **This changes the LSTM-AE architecture and therefore the paper's LSTM
  results.** Set `lstm_frame_size: 1` to reproduce pre-2026-07-18 numbers. It is
  arguably a fairer baseline, since the SNN-AE sees the whole window at once via
  `linear:64` while the old LSTM saw one scalar per step.

Implemented by `to_lstm_frames()` in `src/experiments/meeting01/lib/src/Meeting01Encoding.cpp`;
see [LSTM Performance](../Guides/LSTM-Performance.md) for why a plain reshape
would produce a polyphase split rather than consecutive frames.

### Data Loading Limits

**Superseded — this describes `build_legacy_split`, deleted 2026-09-23 along with the
`article-*.json` profiles that relied on it.** It pooled every loaded window across every
speaker/recording, shuffled the pool (seeded by `experiment.seed`), then sliced train/val
from the shuffled pool with zero group-disjointness — the leakage defect described in the
banner above. `build_split` now unconditionally calls `build_loso_split`
(`Meeting01Dataset.cpp`), which partitions by speaker/recording group per `dataset.cv_fold`;
see [Reviewer-driven revision](#reviewer-driven-revision-meeting01-losojson) for the current
split mechanics. `max_loaded_train_samples`/`max_validation_samples` are still required at
JSON-parse time but are no longer consulted by the split; the active caps are
`loso_max_{train,val,test}_windows`.

### Training Stability and Reproducibility

Because neural network performance can vary based on random weight initialization, the experiment uses a `repeats` parameter to ensure statistical reliability:

1. **Statistical Reliability**: By training each configuration multiple times (e.g., `repeats = 3`), the experiment allows for the calculation of averages and standard deviations, ensuring that results are not due to "lucky" random seeds.
2. **Determinism Verification**: When `seed_deterministic` is enabled, every repeat uses the same seed. The `check_determinism` flag then verifies that the results are identical across all repeats, which is critical for scientific reproducibility.
3. **Stability Analysis**: When `seed_deterministic` is disabled, each repeat uses a unique seed. This reveals how robust the model is to different initializations.

### Profile Configurations

**Superseded — the four `article-*.json` profiles this table listed were deleted
2026-09-23.** `meeting01-loso.json` (nested leave-one-group-out, all 4 datasets, all 4
model families searched by GA) is now the only production profile; see
[Multi-family architecture search](#multi-family-architecture-search-added-2026-09-22-same-day-later-scope-change)
for its shape and `_total_runs_breakdown` field for its actual run/ETA counts.

Profile validation test: `profile_audit_gtest`. Run after every profile edit.

### Dataset Support

**FSDD (Free Spoken Digit Dataset)**:
- Location: `/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset`
- Format: `.wav` audio files (16-bit PCM, mono, 8kHz)
- Organization: `{digit}_{speaker}_{index}.wav`
- Samples: ~3,000 recordings (50 digits × 6 speakers)

### WAV Loading

`to_window_tensor` (windowing/z-score on an already-loaded signal) was deleted
2026-09-23 along with `build_legacy_split`, its only caller; windowing/normalization for
the current LOSO split lives in `build_loso_split` (`Meeting01Dataset.cpp`) instead. The
WAV file reading itself is unaffected — still delegated to the shared FSDD loader:

```cpp
// File: src/core/data_loaders/10.5281/zenodo.1342401/loaders/FsddLoader.cpp
#include "wave/Wav.hpp"

Wav wav;
wav.read(wav_path.string());
const auto& raw = wav.get_data();  // std::vector<double>
if (raw.empty())
    throw std::runtime_error("Empty WAV file: " + wav_path.string());

nn::Tensor signal(static_cast<nn::Index>(raw.size()), 1);
for (std::size_t i = 0; i < raw.size(); ++i)
    signal.at(static_cast<nn::Index>(i), 0) = static_cast<float>(raw[i]);
```

### Progress Tracking

Real-time progress bars during training using `nn::utility::printProgress`:

```cpp
// Inside Meeting01Training.cpp
printProgress(train_samples.size(),
    1,
    train_samples.size() * cfg.epochs,
    epoch * train_samples.size() + train_samples.size(),
    epoch * train_samples.size() + train_samples.size(),
    false,
    run_id,
    total_runs,
    epoch + 1,
    cfg.epochs,
    train_samples.size(),
    train_samples.size(),
    static_cast<double>(val_mse),
    std::span<nn::Tensor*>{},
    "LSTM");
```

Output (multiline ANSI):
```
LSTM Fold:  [===================>                  ]  50%
Epoch: [===================>                  ]  50% (50/100)
Batch: [========================================] 100% (500/500b, 500/500s)  loss: 1.219896
```

### Reconstruction Metrics

The experiment is a **pure reconstruction study**. FSDD carries no anomaly labels, so F1/precision/recall are not computed. Reported metrics:

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | $\frac{1}{N}\sum(x-\hat x)^2$ | Primary quality metric |
| MAE | $\frac{1}{N}\sum|x-\hat x|$ | Robust alternative |
| $R^2$ | $1 - \text{SS\_res}/\text{SS\_tot}$ | Coefficient of determination |
| Spike rate | mean spike fraction over val set | SNN only |
| Energy | $r \cdot N + 10 \cdot \text{MACs}$ (SNN) / $10 \cdot \text{MACs}$ (LSTM) | Proxy; 10× constant = energy cost per MAC vs spike op |

`max_reconstruct_mean_deviation` remains in the config but is only used as a per-sample pass/fail threshold for logging — it does not affect the reported metrics.

### Layer Specification Manual

The architecture of the autoencoder is defined using a domain-specific language (DSL) in the `encoder_layer_spec` and `decoder_layer_spec` lists. Each string in the list represents a layer or a block of layers.

#### 1. Linear Layers
The most common layer used in Experiment04.
- **Format**: `linear:<width>[:<activation>]`
- **Width**: Can be a numeric value (e.g., `32`) or a special token:
    - `hidden`: Resolves to the base hidden size.
    - `latent`: Resolves to the bottleneck dimensionality.
    - `output`: Resolves to the input signal window size.
    - `branch_hidden` / `fusion_hidden`: Resolves to the specific branch/fusion sizes.
- **Activation**: Optional.
    - **ANN Mode**: `relu`, `leaky_relu`, `identity`.
    - **SNN Mode**: `leaky` (LIF), `leaky_integrator`, `identity`.
- **Example**: `linear:64:leaky` $\rightarrow$ A linear layer with 64 units and a LeakyReLU/LIF activation.

#### 2. Convolutional, Pooling, Residual (ANN mode only)
- **Conv1D**: `conv1d:<out_channels>:<kernel_size>[:<stride>[:<activation>]]`
- **Pool1D**: `pool1d:<kernel_size>[:<stride>]`
- **Residual**: `residual` or `residual:<repeat_count>`

> ⚠️ **SNN mode restriction**: In SNN autoencoders (`snn-ae`), `parse_layer_module_spec` only instantiates `linear` entries. Any `conv1d`, `pool1d`, or `residual` entry in `encoder_layer_spec` / `decoder_layer_spec` will **throw a `std::invalid_argument` at startup**. Keep SNN specs to `linear:width:leaky` / `linear:width:identity` only.

#### 3. Standalone Activations
- **Format**: `<activation_type>` (e.g., `relu`)

#### 4. SNN Architecture Modes (input transforms, not network layers)

`snn_architectures` in the profile selects how the raw signal is pre-processed **before** entering the autoencoder. All three modes share the same network (e.g., `linear:64:leaky / linear:32:identity`).

| Mode | Pre-processing applied to input |
|------|-------------------------------|
| `dense` | Pass-through (no transform) |
| `conv1d` | 3-tap smoothing filter: kernel `{0.25, 0.5, 0.25}` |
| `recurrent` | Stand-alone LIF transform (stateless, fixed V_th/alpha) |

These are **not** different network architectures — they are signal conditioning steps applied at `Meeting01Encoding.cpp:apply_snn_architecture_transform`.

The `recurrent` transform iterates the window samples as time steps:
$v[t] = \alpha\,v[t-1] + x[t] - s[t-1]\,V_{th}$, $s[t] = \mathbb{1}[v[t] \ge V_{th}]$.
It normally returns only the spike train $s$; `meeting01::recurrent_lif_trace` (bound as
`nn_microscope.meeting01.recurrent_lif_trace`) returns the discarded membrane trajectory
$v[t]$ as well, so the [Experiment Microscope](../Guides/Experiment-Microscope.md) SNN Lab
can plot a real per-step membrane curve. This is the only place meeting01 has a membrane
*trajectory* — the SNN-AE itself runs `time_steps == 1` (the window is a feature vector,
so its LIF layer has a single `v_mem` per neuron, not a curve).

#### Building a Full Architecture
The total network is built by concatenating these specs. 
**Example Encoder**: `["linear:128:leaky", "residual:2", "linear:32:identity"]`
1. Linear(input $\rightarrow$ 128) $\rightarrow$ Lif ReLU
2. 2x Residual Blocks (128 $\rightarrow$ 128)
3. Linear(128 $\rightarrow$ 32) $\rightarrow$ Identity (Latent Bottleneck)

## Usage

```bash
# Run a single (dataset, fold) slice directly (from software/nn/)
./out/build/max-performance/src/experiments/meeting01/meeting01 \
  --comparative-config src/experiments/meeting01/profiles/meeting01-loso.json \
  --dataset fsdd --cv-fold 0

# Run the full nested-LOSO grid (all datasets x folds) + build paper CSVs
# (weeks-scale; see Re-run Runbook)
EXPERIMENT_CONFIRMED=1 ./scripts/pipeline/meeting01/01_meeting01_run_loso.sh
```

Both `--comparative-config` and `--profile` are accepted as the flag name.

### Outputs

Results written to `results/` (or `dataset.results_dir` from profile):

| File | Contents |
|------|----------|
| `{run_tag}_comparative_metrics.csv` | One row per (model, encoding, architecture, v_th, alpha, run_id) |
| `{run_tag}_publication_table.csv` | Aggregated, formatted for paper tables |
| `{run_tag}_summary.json` | Config hash, per-model stats |
| `data/{run_tag}_*.dat` | pgfplots DAT files for paper figures |
| `data/paper_*.csv` | Aggregated across all runs (written by `02_meeting01_build_lstm_vs_snn_paper_data.py`) |

Checkpoints in `results/checkpoints/` — safe to interrupt and resume.

#### Saved models (`dataset.save_models: true`)

Written to `results/meeting01/models/`, per (dataset, fold, encoding, architecture,
v_th, alpha, run):

| File | Format | Consumer |
|------|--------|----------|
| `…_encoder_params.txt` / `…_decoder_params.txt` | text parameter dump | human inspection, `scripts/data/npz_to_pytorch.py` |
| `…_encoder.npz` / `…_decoder.npz` | `NetworkSerializer` npz (`Linear`/`Lif`) | `nn_microscope.meeting01.snn_ae_forward` — the [Experiment Microscope](../Guides/Experiment-Microscope.md) reloads these to reproduce a window's latent + reconstruction without retraining |

`role` in the filename is `combo` for a GA-evaluated candidate genome and `final` for
the retrained winner. `01_meeting01_run_loso.sh` clears `results/meeting01/models/`
on a fresh run.

### Paper data pipeline

`01_meeting01_run_loso.sh` chains into this automatically once the grid finishes (skip with
`SKIP_POSTPROCESS=1` and run manually later — see [Re-run Runbook](../Guides/Re-run-Runbook.md)):

```bash
python3 scripts/pipeline/meeting01/03_meeting01_pca_mean_baselines.py       # --results-dir/--run-tag default to results/meeting01, meeting01_loso
python3 scripts/pipeline/meeting01/02_meeting01_build_loso_paper_data.py    # writes DAT files to --data-dir (defaults to documentation/07-articlesProduced/meeting01/data)
python3 scripts/pipeline/meeting01/04_meeting01_significance_tests.py

# Compile paper:
cd documentation/07-articlesProduced/meeting01
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Key Differences from AutoencoderRunner

| Feature | AutoencoderRunner | Experiment04 |
|---------|-------------|--------------|
| Model | Feedforward AE | LSTM + SNN comparative |
| Input | Fixed-dim vectors | Variable-length sequences |
| Latent | Vector | Final hidden state |
| BPTT | Not used | LSTM uses BPTT |
| Dataset | 10.1117 EEG/Audio | FSDD (spoken digits) |
| Progress | Legacy async | nn::progress (ANSI) |

## Common Pitfalls

1. **SNN spec with non-linear entries throws at startup.** `parse_layer_module_spec` in SNN mode only handles `linear:width[:activation]`. Any `conv1d:`, `pool1d:`, `residual`, or `spiking_neuron:` entry throws `std::invalid_argument` before training begins.

2. **`early_stop_patience` must be < `epochs`.** `validate()` enforces this — profile will reject with a clear error if violated.

3. **`seed_deterministic: true` with `repeats > 1` produces identical runs.** Use `false` for article profiles (different seed per repeat).

4. **SNN architecture modes are signal transforms, not layers.** `conv1d`/`recurrent` in `snn_architectures` do not add conv or LSTM layers to the network — they pre-process the input window before it enters the autoencoder.

5. **FSDD path must match `dataset_root` in profile.** Default: `/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset`.

6. **F1/precision/recall are always 0 for FSDD.** FSDD has no anomaly labels. These fields exist in the output CSV but should not be cited.

7. **Benchmark runs resume from `results/checkpoints/`.** Results are cached by config hash, so re-running after a code change reuses the old numbers — a "run" that finishes in seconds with metrics identical to the previous one is the tell. Delete the results directory before any timing comparison.

8. **`lstm_frame_size` changes LSTM-AE results, not just its speed.** Do not mix runs with different values in one comparison table.

## Results

> **Grid-era results, pending a GA re-run.** Everything below was collected while the
> SNN arm still used the fixed `linear:64:leaky, linear:32:identity` shape selected by
> the exhaustive `v_th × alpha × architecture` grid (see
> [NSGA-II Architecture Search](#nsga-ii-architecture-search-added-2026-09-22-grid-removed-2026-09-22)
> for what replaced it). The grid code no longer exists, so these numbers cannot be
> reproduced by re-running the current binary — they describe the files already on disk
> in `results/`, not the current search mechanism. Row counts (e.g. "81 rows") describe
> those existing CSVs, not what a fresh GA run would produce (~144 rows at the current
> population/generations/seeds budget). Treat this whole section as a snapshot to be
> superseded once a real GA run completes.
>
> **They are also pre-`T=16`.** These SNN numbers were produced at `time_steps = 1`, with
> no membrane dynamics, an inert `alpha`, an unset `voltage_threshold`, a latency encoding
> emitting ~1 spike per 256-sample window, and an encoding-dependent reconstruction target.
> See [The Missing Time Axis](#the-missing-time-axis-found--fixed-2026-09-22-second-pre-gridunesp-audit).
> They do not describe a weaker version of the current SNN — they describe a different
> model.

All results from 3 independent runs, FSDD dataset, window size 256, Adam(lr=1e-3, β₁=0.9, β₂=0.999), up to 30 epochs with early stopping (patience=10). SNN: 2 linear layers (64→32 latent). LSTM: 1-layer hidden=64, latent=32.

### Raw Publication Tables

Values are means across 3 runs. Energy unit: proxy score = spike\_rate × N + 10 × MACs (dimensionless relative measure). Train time in ms.

#### LSTM-AE (article-lstm-ae profile, 3 repeats)

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9978 | 0.7049 | 0.0022 | 0 | 8.52e+07 | 452 708 |
| lstm-ae | latency | 1 | 0.2348 | 0.4780 | 0.0483 | 0 | 8.52e+07 | 186 683 |
| lstm-ae | poisson | 1 | 0.1179 | 0.2523 | −0.0352 | 0 | 8.52e+07 | 158 527 |

#### SNN-dense (article-snn-dense profile, 3 repeats)

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9968 | 0.7054 | 0.0032 | 0 | 8.52e+07 | 666 860 |
| lstm-ae | latency | 1 | 0.2415 | 0.4863 | 0.0213 | 0 | 8.52e+07 | 385 226 |
| lstm-ae | poisson | 1 | 0.1185 | 0.2728 | −0.0408 | 0 | 8.52e+07 | 572 979 |
| snn-ae | direct | 2 | 1.1056 | 0.7709 | −0.1056 | 0.499 | 372 471 | 3 986 |
| snn-ae | latency | 2 | 0.1364 | 0.2673 | 0.4472 | 0.877 | 375 374 | 3 266 |
| snn-ae | poisson | 2 | 0.1244 | 0.1707 | −0.0921 | 0.970 | 376 092 | 6 485 |

#### SNN-conv1d (article-snn-conv1d profile, 3 repeats)

Conv1d mode = 3-tap smoothing filter {0.25, 0.5, 0.25} applied before encoding.

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9968 | 0.7054 | 0.0032 | 0 | 8.52e+07 | 302 894 |
| lstm-ae | latency | 1 | 0.2415 | 0.4863 | 0.0213 | 0 | 8.52e+07 | 323 118 |
| lstm-ae | poisson | 1 | 0.1185 | 0.2728 | −0.0408 | 0 | 8.52e+07 | 281 009 |
| snn-ae | direct | 2 | 0.8052 | 0.6342 | −0.1155 | 0.489 | 372 393 | 1 963 |
| snn-ae | latency | 2 | 0.1032 | 0.2331 | 0.5269 | 0.881 | 375 410 | 2 809 |
| snn-ae | poisson | 2 | 0.0697 | 0.1516 | −0.1284 | 0.994 | 376 271 | 1 363 |

#### SNN-recurrent (article-snn-recurrent profile, 3 repeats)

Recurrent mode = stand-alone LIF transform on input before encoding.

| Model | Encoding | Layers | MSE | MAE | R² | Spike Rate | Energy | Train ms |
|-------|----------|--------|-----|-----|----|------------|--------|----------|
| lstm-ae | direct | 1 | 0.9968 | 0.7054 | 0.0032 | 0 | 8.52e+07 | 318 625 |
| lstm-ae | latency | 1 | 0.2415 | 0.4863 | 0.0213 | 0 | 8.52e+07 | 566 580 |
| lstm-ae | poisson | 1 | 0.1185 | 0.2728 | −0.0408 | 0 | 8.52e+07 | 435 531 |
| snn-ae | direct | 2 | 0.1276 | 0.1786 | −0.0876 | 0.947 | 375 917 | 8 912 |
| snn-ae | latency | 2 | 0.1594 | 0.2859 | 0.2507 | 0.834 | 375 043 | 18 182 |
| snn-ae | poisson | 2 | 0.1156 | 0.1605 | −0.0836 | 0.970 | 376 090 | 15 787 |

---

### Compiled Analysis

#### Best MSE per encoding (lowest = best reconstruction)

| Encoding | Best model | MSE | vs LSTM-AE MSE | Improvement |
|----------|-----------|-----|----------------|-------------|
| direct | SNN-recurrent | 0.1276 | 0.9978 | **7.8×** |
| latency | SNN-conv1d | 0.1032 | 0.2348 | **2.3×** |
| poisson | SNN-conv1d | 0.0697 | 0.1179 | **1.7×** |

SNN consistently outperforms LSTM-AE on reconstruction MSE for all three encodings. Best overall: SNN-conv1d + poisson (MSE = 0.0697).

#### Energy efficiency (proxy: spike\_rate × N + 10 × MACs)

| Model | Energy (mean) | vs LSTM-AE | Factor |
|-------|--------------|------------|--------|
| LSTM-AE | ~8.52 × 10⁷ | baseline | 1× |
| SNN-dense | ~374 646 | 227× lower | **227×** |
| SNN-conv1d | ~374 691 | 227× lower | **227×** |
| SNN-recurrent | ~375 683 | 227× lower | **227×** |

SNN architecture mode does not affect energy — all three use the same network; the pre-processing transform is essentially free.

#### Spike rates

| Architecture | Encoding | Spike Rate |
|-------------|----------|------------|
| dense | poisson | 0.970 |
| conv1d | poisson | 0.994 |
| recurrent | poisson | 0.970 |
| dense | latency | 0.877 |
| conv1d | latency | 0.881 |
| recurrent | latency | 0.834 |
| dense | direct | 0.499 |
| conv1d | direct | 0.489 |
| recurrent | direct | 0.947 |

Poisson encoding produces highest spike rates (~0.97–0.99). Direct encoding varies by architecture (recurrent: 0.947, dense/conv1d: ~0.49).

#### Training time

| Model | Fastest encoding | Time (ms) | Slowest encoding | Time (ms) |
|-------|-----------------|-----------|-----------------|-----------|
| LSTM-AE | poisson | 158 527 | direct | 452 708 |
| SNN-dense | latency | 3 266 | poisson | 6 485 |
| SNN-conv1d | poisson | 1 363 | latency | 2 809 |
| SNN-recurrent | direct | 8 912 | latency | 18 182 |

SNN-conv1d trains fastest (1 363–2 809 ms). SNN-recurrent is slowest among SNNs (~9–18 s) due to LIF transform overhead per window. LSTM-AE trains slowest overall (~158–453 s).

#### R² summary

Positive R² indicates the model explains variance beyond the mean baseline. Only some configurations achieve positive R²:

| Model | Encoding | R² |
|-------|----------|----|
| SNN-conv1d | latency | **0.527** |
| SNN-dense | latency | **0.447** |
| SNN-recurrent | latency | **0.251** |
| LSTM-AE | latency | 0.048 |
| LSTM-AE | direct | 0.002–0.003 |
| all others | direct/poisson | < 0 |

Latency encoding is the only configuration where models learn meaningful variance structure. All direct and poisson encodings yield near-zero or negative R², indicating the models learn the signal mean but not its shape.

---

### Raw Data Files

| File | Description |
|------|-------------|
| `results/article_lstm_ae_comparative_metrics.csv` | Per-run raw metrics (9 rows: 3 encodings × 3 runs) |
| `results/article_snn_dense_comparative_metrics.csv` | Per-run raw metrics (81 rows: 3 enc × 3 arch-sweep × v_th × alpha × 3 runs) |
| `results/article_snn_conv1d_comparative_metrics.csv` | Same structure as dense |
| `results/article_snn_recurrent_comparative_metrics.csv` | Same structure as dense |
| `results/article_*_publication_table.csv` | Aggregated mean over runs, formatted for paper |
| `results/article_*_summary.json` | Config hash, seed, stat tests |
| `data/article_*_history.dat` | Per-run epoch loss curves (pgfplots format) |
| `data/article_*_convergence.dat` | Convergence diagnostic per run |
| `data/article_*_summary.dat` | Summary statistics (pgfplots format) |
| `data/article_*_sweep.dat` | Hyperparameter sweep results |

## NSGA-II Architecture Search (added 2026-09-22, grid removed 2026-09-22)

**Problem.** The SNN-AE's own shape was never searched — `encoder_layer_spec`/
`decoder_layer_spec` were a profile constant (`linear:64:leaky, linear:32:identity`),
dimension-matched to the LSTM/GRU/Transformer baselines. Only `architecture` (the
input-transform selector: dense/conv1d/recurrent), `v_th`, and `alpha` were swept, via
an exhaustive 3×3×3=27-candidate grid (`run_snn_sweep`, one grid per encoding × dataset
× fold × seed).

**What changed.** NSGA-II search now evolves the SNN-AE's architecture jointly with its
hyperparameters. **This is the only SNN architecture search mechanism in the code —
`run_snn_sweep`, `run_snn_combo`, and the `v_th_values`/`alpha_values` discrete-sweep
config fields were deleted outright, not kept behind a toggle.** The initial
implementation added GA as an opt-in `evaluation.ga.enabled` flag alongside the
unchanged grid; this was deliberately reversed the same day at the user's explicit
request, specifically to rule out any code path that could produce an SNN result not
generated by the GA. There is no default that silently falls back to a grid — a profile
either declares `evaluation.snn_architectures` non-empty (GA runs) or empty (no SNN
arm), full stop.

| Gene | Range | Notes |
|---|---|---|
| `encoder_widths` | free length + free per-element width | layer count AND neurons/layer are both evolved (strictly decreasing, last = latent); decoder mirrors the encoder |
| `encoding` | direct / poisson / latency | drawn from `evaluation.encodings` (the profile's existing whitelist) |
| `architecture` | dense / conv1d / recurrent | drawn from `evaluation.snn_architectures` (existing whitelist) |
| `voltage_threshold`, `alpha` | `evaluation.ga.{voltage_threshold,alpha}_{min,max}` | continuous, no longer discrete sweep lists |

Objectives (minimized): validation MSE, and inference cost (`estimate_snn_macs`,
reused unchanged — `Meeting01Metrics.hpp`). No feasibility constraint is modeled: unlike
paraconsistentGA's latent-collapse guard (a real, previously observed failure mode of
*that* pipeline's scoring), plain reconstruction MSE has no known degenerate false
optimum — a collapsed model scores worse, not better — so every individual is
unconditionally feasible and NSGA-II's constrained-dominance branch degenerates to
plain Pareto dominance. Haploid, not diploid: the diploidy paraconsistentGA uses exists
specifically to hedge against `d_penalized`'s known false optimum (the Ambiguity
vertex); that rationale does not transfer here.

Budget: **population=8, generations=5** in the shipped production profiles, μ+λ elitism,
48 evaluations per (dataset, fold, seed), plus `|Pareto front| × (winner_seeds − 1)`
re-scorings of the final front. This was cut from the originally planned 10×9=90 on
2026-09-22 to help pay for real temporal simulation (`time_steps = 16`) — see
[The Missing Time Axis](#the-missing-time-axis-found--fixed-2026-09-22-second-pre-gridunesp-audit)
for the full compensation table. Every SNN-bearing profile shipped in `profiles/` declares
`evaluation.ga` explicitly (population/generations/`winner_seeds`/`voltage_threshold_min-max`/
`alpha_min-max`) rather than relying on the invisible struct default, so the search budget
for a paper run is readable straight from the profile. Because `encoding` moved from an
outer sweep loop into the genome, one GA run covers all three encodings jointly — removing
the ×3 outer encoding loop for the SNN arm is what made the reduced budget affordable
without shrinking the searched space.

**Consequence for results framing (real, not a bug).** The old grid's `evaluation.encodings`
loop applied the *same* encoding to the SNN **and** the baselines for each of 3 rows —
"SNN vs LSTM under latency encoding" was a controlled, matched-input comparison. Now that the
SNN's own encoding is GA-evolved (picked once per fold, not fixed per row), the SNN no
longer has a separate number *under* each encoding — one number per fold, using
whichever encoding the GA found best. Baselines still run per encoding (`run_baseline_family`
is unchanged). So the comparison is now "GA-optimized SNN (its own preferred encoding
logged per fold) vs. LSTM/GRU/Transformer under each of 3 controlled encodings," not
"matched-encoding SNN vs LSTM/GRU/Transformer." The `## Results` R²
table above (`Latency encoding is the only configuration where models learn meaningful
variance structure`) is a grid-search-era, matched-encoding finding from before this
change — it does not carry over as a like-for-like SNN-side result; the SNN-side
analogue is the GA's own encoding-selection frequency, logged per individual in
`*_model_selection_manifest.json`. This table needs updating once a real GA run exists.

**Where it lives:**

| File | Role |
|---|---|
| `lib/include/Meeting01GaGenome.hpp` + `.cpp` | Genome struct, repair/random/crossover/mutate, `to_ae_config` |
| `lib/include/Meeting01GaFitness.hpp` + `.cpp` | `Meeting01GaIndividual`, `evaluate_individual` (trains + scores one genome) |
| `lib/include/Meeting01GaSearch.hpp` + `.cpp` | The NSGA-II generational loop, `pick_winner` |
| `lib/include/Meeting01GaCheckpoint.hpp` + `.cpp` | Two-layer crash-resilient checkpoint (per-individual cache + per-generation state), ported from `paraconsistentGA`'s `GaCheckpoint` |
| `include/ga/Nsga2Core.hpp` | Shared, population-shape-agnostic NSGA-II core (`constrained_dominates`/`fast_non_dominated_sort`/`assign_crowding_distance`), used by both `paraconsistentGA` and this search — see [ParaconsistentGA](ParaconsistentGA.md) |
| `Meeting01Experiment.cpp`: `finalize_snn_selection` | The nested-LOSO final retrain-on-train∪val + test-evaluation tail: retrains the GA winner from `run_snn_ga_search`, early-stops on the carved recording-disjoint monitor, evaluates once on the held-out test speaker |
| `Meeting01Experiment.cpp`: `run_snn_ga_search` | Builds the initial population, runs the generational NSGA-II loop, picks the winning genome from the rank-0 Pareto front — the only entry point into the SNN arm |

Tests: `tests/meeting01_ga_gtest.cpp` — genome legality, the free-form `to_ae_config`
path reproducing today's fixed profile shape as a regression anchor, checkpoint
round-trip, the shared NSGA-II core contract, and (added 2026-09-22) the selection
machinery itself: `pick_winner`'s (val_mse, then cost) ordering and its empty-front throw,
non-dominated sort separating dominated individuals, crowding preserving the front's
boundary solutions, the `population_size = 1 with generations ≥ 1` validation guard that
replaced an infinite loop, and the MAC proxy distinguishing `{128, 8}` from `{128, 120}`.
Paper text changes (Limitations / Methodology) are deliberately **not** part of this
change — they need a real GA run's results to write accurately.

## Multi-family architecture search (added 2026-09-22, same day, later scope change)

### The question changed, not just the code

Everything in the section above answers "does a *searched* SNN beat three *fixed*
baselines?" — LSTM-AE, GRU-AE and Transformer-AE each still had exactly one shape
(`hidden_size=64`, `d_model=64`, …), read straight from the profile, never evolved.
That comparison is unfair in a specific, nameable way: the SNN got to try dozens of
shapes and report its best; the baselines got exactly one guess each. A baseline that
lost might just have had the wrong `hidden_size` — nothing in the old design could tell
you which.

The user's decision (2026-09-22, same day as the section above): stop asking "SNN vs.
three fixed baselines" and start asking **"what is the best autoencoder overall, at a
fixed compression ratio, for the next thesis phase and a paper"** — which means every
family now gets the same deal the SNN already had: its own architecture, searched.

### Four searches, not one population

A tempting shortcut would be one NSGA-II population containing all four model types,
competing directly. That was considered and rejected (user decision, locked via
`AskUserQuestion`): a mixed population needs crossover between an SNN genome and a
Transformer genome to mean something, and it doesn't — there is no principled way to
"average" `encoder_widths=[96,40,12]` with `d_model=64,n_heads=4`. Instead:

```
   SNN-AE search  --> Pareto front --> winner_snn  (val_mse, cost)
   LSTM-AE search --> Pareto front --> winner_lstm (val_mse, cost)
   GRU-AE search  --> Pareto front --> winner_gru  (val_mse, cost)
   Transformer-AE search --> Pareto front --> winner_transformer (val_mse, cost)
                                    |
                                    v
                    min(val_mse, then cost) over the 4 winners
                                    |
                                    v
                    <run>_overall_winner_manifest.json
```

Four independent NSGA-II runs, each with its own population, its own Pareto front, its
own winner — then the 4 winners (not the 4 populations) are compared once, by the exact
same rule `pick_winner` already used inside each search: lowest validation MSE, ties
broken by lowest inference cost. This is the same shape as choosing a race winner per
event and then comparing gold-medal times across events, not merging every runner from
every event into one race.

### `latent_dim` is fixed, not evolved — on purpose

Every genome (SNN, recurrent, Transformer) can freely evolve its hidden width, depth,
attention heads, and input encoding. None of them can evolve `latent_dim`. It is read
once from `cfg.model.latent_dim` and held fixed across all four families and every
individual in every population, for that dataset's runs.

**Why this matters:** the bottleneck width is *the* thing that decides how hard the
compression problem is. An SNN allowed a 64-wide bottleneck and a Transformer stuck at
8 are not competing on architecture quality — the SNN's task is strictly easier. Fixing
`latent_dim` means a win is "this family reconstructs better *through the same size
hole*," not "this family gave itself a bigger hole." Every genome struct's bounds
comment says this explicitly (`GenomeBounds::latent_dim`, `RecurrentGenomeBounds`'s
header note, `TransformerGenome`'s header note) — the same sentence, repeated at each
of the four call sites, so nobody wiring a fifth family later has to go hunting for the
rule.

**The hole is per dataset, not a single global number (2026-09-23).** Before this date
`latent_dim` was one value (32) shared by all 4 datasets, with no recorded justification
— `git log -S'latent_dim'` traces it to the prototype's earliest commits with nothing
explaining the choice. Since meeting01 is part of a thesis that explicitly compares
handcrafted vs. automated feature-engineering strategies, an unexamined magic number
here was the wrong default to keep. A short literature pass (see table) found that audio
and EEG autoencoders use different bottleneck widths in practice, so `latent_dim` is now
per **signal domain**, resolved once per dataset before that dataset's 4 searches start —
still fixed *within* a dataset (the "same hole" argument above is untouched), just no
longer fixed *across* dataset domains that were never comparable to begin with.

| Dataset | Domain | `latent_dim` | Ratio (window=256) | Source |
|---|---|---|---|---|
| fsdd | short speech | 16 | 16:1 | Traditional/LSTM speech autoencoders typically bottleneck at 8–16. |
| audiomnist | short speech | 16 | 16:1 | Same domain as fsdd; AudioMNIST encoder architectures in the literature sit in the 16–64 range. |
| eegmmidb | EEG (motor imagery) | 64 | 4:1 | No eegmmidb-specific autoencoder paper found; extrapolated from Khan et al. (2023)'s shallow-AE result on CHB-MIT (same EEG modality and window scale, but a dataset not used in this study — see `siena` row) — unverified extrapolation, not a direct citation. |
| siena | EEG (seizure) | 64 | 4:1 | No Siena-specific autoencoder paper found; extrapolated from Khan et al. (2023), shallow autoencoder on CHB-MIT, hidden_size=64 (a separate {32, 64, 128} sweep on CHB-MIT also selected 64) — same seizure-EEG task category as Siena, not a direct hit. Siena replaced CHB-MIT itself in the active grid 2026-09-23 (CHB-MIT's 42.6GB did not fit available disk; Siena is a complete 14-subject, 20.3GB dataset in the same category). |

A tempting-looking counter-claim surfaced during the search — "latent dimensions ≤50
fail to reconstruct reliably" — and was checked all the way to the source (arXiv:2109.11045)
rather than taken at face value: it is specific to **MNIST images (784-dim)**, holds only
for the paper's **non-spiking** baselines, and that paper's own **spiking autoencoders are
the stated exception**. It does not transfer to meeting01's SNN-AE or to a 256-sample 1-D
signal window, and was discarded as a reason to keep a larger `latent_dim`.

Mechanism: `Meeting01Config::Dataset::resolve(name)` returns a `DatasetSource` whose
`latent_dim` follows the same "0 inherits, >0 overrides" rule already used for
`sample_rate`/`max_windows_per_recording` — a `DatasetSource`-level value wins if set,
else the `Dataset`-level value, else 0. Inside `run_comparative_experiment`
(`Meeting01Experiment.cpp`), each iteration of the per-dataset loop resolves this value
and, only when it is `> 0`, copies the whole config with `model.latent_dim` overridden
for that iteration — a profile that never sets a per-dataset `latent_dim` keeps reading
its own single global `model.latent_dim`, unchanged from today's behavior.

**A "same hole" hole, found and closed the same day.** The "same hole" guarantee above
only held structurally for the SNN: `repair_widths` forces `hidden_min =
max(min_width, latent_dim+1)`, so every SNN genome's real bottleneck is exactly
`latent_dim`. LSTM-AE/GRU-AE and Transformer-AE had no equivalent floor —
`repair_recurrent`/`repair_transformer` clamp `hidden_size`/`d_model` only to
`[min_hidden, max_hidden]`/`[min_d_model, max_d_model]`, never checking `latent_dim`.
Both models project down to the latent through a separate linear
(`LSTMAutoencoder::enc_proj_: H→Z`, `TransformerAutoencoder::to_latent_: d_model→Z`), so
a genome drawing `hidden_size`/`d_model` below `latent_dim` has a REAL bottleneck
narrower than the one the comparison assumes — silently, no crash, just a worse-than-
labeled individual. With a single global `latent_dim=32` this affected a modest slice of
the search range (`min_hidden=8` → 24/249 ≈ 9.6% of draws for LSTM/GRU, `min_d_model=16`
→ 16/113 ≈ 14.2% for Transformer). Making `latent_dim` per-dataset made it worse
specifically for EEG (`latent_dim=64`): 56/249 ≈ 22.5% for LSTM/GRU, 48/113 ≈ 42.5% for
Transformer — found during the per-dataset `latent_dim` review, before any real EEG data
was on disk to run against, and fixed the same day: `run_lstm_ga_search`/
`run_gru_ga_search`/`run_transformer_ga_search` now set `bounds.min_hidden`/
`bounds.min_d_model = std::max(profile_value, config.model.latent_dim)`, mirroring the
SNN's own floor. The profile's own `min_hidden`/`min_d_model` values are untouched —
this only raises the *effective* floor for datasets whose `latent_dim` exceeds it.

### `d_model % n_heads == 0`: the one constraint no other family has

Every other gene in every other genome is a free integer inside a range — any
`hidden_size` in `[min_hidden, max_hidden]` is legal. The Transformer's `d_model` is
not: multi-head attention splits the embedding into `n_heads` equal pieces, so
`d_model` must be an exact multiple of `n_heads`, not merely close to one. `d_model=100,
n_heads=3` is not a worse Transformer — it is a Transformer that cannot be constructed
at all.

`repair_transformer` (`Meeting01TransformerGaGenome.cpp`) enforces this the same way
`repair_widths` enforces the SNN's strictly-decreasing widths: applied unconditionally
after every random draw, crossover, and mutation, never left to chance. It first snaps
`n_heads` to the nearest value in `head_choices` (a short list of legal counts — 1, 2,
4, 8 — not an arbitrary range, because "nearest integer" doesn't make sense for a divisor
constraint the way it does for a width), then rounds `d_model` to the nearest legal
multiple of *that* `n_heads` inside `[min_d_model, max_d_model]`. `meeting01_transformer_ga_gtest.cpp`
asserts `d_model % n_heads == 0` after 200 random draws, 100 crossovers, and 500
mutations, plus a direct repair of a deliberately illegal `(d_model=100, n_heads=3)`
pair — the one property that must never slip.

### Distinct C++ types for one shared genome (the LSTM/GRU trick)

LSTM-AE and GRU-AE vary over exactly the same axes — `hidden_size`, `num_layers`,
`encoding` — so they share one genome struct, `RecurrentGenome`. But the generic search
driver (`run_ga_search<Ind>`, templatized once and reused by all four families) needs to
know, for a given individual, which recurrent cell to actually train — and that
decision can't be a runtime string threaded through every call site without undoing the
point of having a generic driver.

The resolution: `LstmGaIndividual` and `GruGaIndividual` are two **empty structs**, both
deriving from `GaIndividualT<RecurrentGenome>`:

```cpp
struct LstmGaIndividual : GaIndividualT<RecurrentGenome> {};
struct GruGaIndividual  : GaIndividualT<RecurrentGenome> {};
```

Same layout, same genome, same everything — except the C++ type. `evaluate_individual`
is overloaded on that type, so `void evaluate_individual(LstmGaIndividual&, ...)` trains
an LSTM and `void evaluate_individual(GruGaIndividual&, ...)` trains a GRU, and ordinary
overload resolution — not a family string checked at runtime — decides which one runs.
The generic driver never special-cases either type; it just calls `evaluate_individual`
unqualified (ADL) on whatever `Ind` it was instantiated with. The Transformer, having no
sibling to share a genome with, doesn't need this trick: `TransformerGaIndividual` is a
plain alias, `GaIndividualT<TransformerGenome>`.

### Equal TIME budget, not equal evaluation count

**The problem this solves.** "Give every family the same population size and generation
count" sounds fair and is not: population×generations counts *evaluations*, and one
evaluation costs wildly different amounts of wall-clock time per family, because it costs
a different number of *sequential steps*. The encoder turns every window into a
`(time_steps, window_size)` tensor before any family-specific framing — that axis is
real and shared, not an SNN-only artifact — but each family then unrolls a different
number of steps over it:

| Family | Steps actually unrolled | Why |
|---|---|---|
| SNN-AE | `time_steps` (16 in production) | LIF membrane updates once per simulation step |
| LSTM-AE / GRU-AE | `(window_size * time_steps) / lstm_frame_size` (512 at `window_size=256, time_steps=16, lstm_frame_size=8`) | recurrent cell processes one frame per step, sequentially, no cross-step batching |
| Transformer-AE | same 512, but attention is over the whole sequence per layer, not one step at a time | self-attention is `O(seq_len²)` per layer, not `O(seq_len)` |

512 sequential recurrent steps per sample vs. 16 membrane updates is not a rounding
difference — equal *evaluation counts* across families would have handed the SNN a
generous search budget and starved the other three, or handed the SNN a starved budget
to let the others run 32× more evaluations, either way silently. This is exactly why
"48 evaluations for everyone" was rejected in favor of measuring real wall-clock cost per
family and solving for the evaluation count each one can afford in the SNN's already-
budgeted time.

**The method.** SNN's budget was fixed first (population=8, generations=5 → 48
evaluations, see the section above) and its wall-clock cost at production settings (200
train windows, 150 validation windows, `time_steps=16`, `window_size=256`) becomes the
TIME ceiling every other family must fit inside. A short probe (population=1,
generations=0 — exactly one evaluation — for each family, pinned to a
production-representative genome: the same `H=64`/`d_model=64,n_heads=4,n_layers=2,d_ff=128`
dimensions the old fixed baselines used) measures real seconds-per-epoch at those exact
settings. Assuming the same expected epochs-per-evaluation across families — a stated,
checkable assumption, not a hidden one: all four share the identical `epochs=30`/
`early_stop_patience=5`/learning-rate regime and train on the same windows, so nothing
in the setup would make one family's early-stopping behavior systematically different
from another's without actually observing it — the equal-time condition reduces to:

```
population_family × (1 + generations_family)   sec_per_epoch_SNN
──────────────────────────────────────────── = ──────────────────
population_SNN × (1 + generations_SNN)          sec_per_epoch_family
```

**The measurement, and the wall it hit.** A probe (population=1, generations=0, one
evaluation, production-representative genome per family) at production settings
(window_size=256, time_steps=16, 200 train / 150 val windows) measured:

| Family | sec/epoch (measured) | vs. SNN |
|---|---|---|
| SNN | 0.5 | 1× |
| Transformer | 83.75 | 168× |
| LSTM | 97.2 | 194× |
| GRU | 136.0 | 272× |

SNN's already-fixed budget (48 evaluations) costs **at most ~12 minutes** even at the
full 30-epoch cap. At 168–272× that per-epoch cost, 12 minutes buys the other three
families **less than one evaluation each** — the equal-time design, as originally
conceived, is mathematically incompatible with running any real search on LSTM/GRU/
Transformer. This was not visible before the measurement: the plan's original "roughly
quadruples total wall-clock" estimate (see [the Cost consequence section](#cost-consequence-stated-plainly-not-buried))
assumed the baselines would cost about the same per evaluation as the SNN. They do not.

**Resolution (user decision, 2026-09-23, confronted with the exact numbers above).**
Equal wall-clock time was abandoned for the three baseline families in favor of each
getting its **own independently-sized budget**. Offered a smaller, cheaper option
(`population=4, generations=4`, ~29 evaluations worst case, ≈ 288 days grid-wide) against
a larger one matching the SNN's own shape, the user edited `meeting01-loso.json`
directly to set `ga.lstm`/`ga.gru`/`ga.transformer` to **`population=8, generations=5,
winner_seeds=3` — the same shape as `ga.snn`** (48 search evaluations each, ~65 with
re-scoring and the final retrain) — deliberately choosing search thoroughness over
minimizing wall-clock, consistent with a standing preference for the more rigorous
option even at real cost. Worst-case total per (dataset, fold, run) cell, assuming every
evaluation runs the full 30-epoch cap (no early stop): SNN ≈ 0.27 h, Transformer ≈ 45.4 h,
LSTM ≈ 52.7 h, GRU ≈ 73.7 h — summing to **≈ 172 h/cell**. Across the grid as it stood
that day (3 datasets × 6 folds × 5 repeats = 90 cells): **≈ 15 476 CPU-hours, ≈ 645 days
(≈ 21.5 months) worst case.** The grid widened to 4 datasets the same day (`mitbih`
replaced by two EEG datasets — see [Three datasets](#three-datasets-multi-dataset-answer-to-one-small-database));
per-cell cost is unchanged, so this scales directly: **120 cells → ≈ 20 635 CPU-hours,
≈ 860 days (≈ 28.7 months) worst case** (cited from the profile's own
`_total_runs_breakdown`, not re-derived here). Real elapsed time will be lower — `early_stop_patience=5` truncates
most evaluations well before 30 epochs — but no measured average-epochs-under-early-
stopping exists, so this is a stated **upper bound**, not a point estimate; deriving a
tighter number would need an actual multi-evaluation run, which is exactly the
multi-month commitment being sized here. The one OS process per (dataset, fold) design
already lets cells run in parallel across a cluster to reduce *elapsed* wall-clock — CPU-
hours are unchanged. `meeting01-loso.json`'s own `_total_runs_breakdown` carries this
same derivation so the number is checkable against the profile that will actually run,
not just this page.

**Failure mode this prevents, and how loud it is.** Before this fix, `meeting01-loso.json`
shipped without `ga.lstm`/`ga.gru`/`ga.transformer` set at all, silently falling back to
`Meeting01Config`'s struct defaults (population=10, generations=8 → 90 evaluations) for
all three — a number picked for no reason connected to this profile's actual cost, just
however many evaluations the SNN's own struct happened to default to before its budget
was calibrated. That failure is **silent**: the run completes, produces plausible
numbers, and nothing anywhere says the three baseline arms cost roughly 90/48 ≈ 1.9× the
SNN arm each rather than the intended equal-time budget — and even that 1.9× framing
turns out to have been the wrong order of magnitude entirely once the real per-epoch
cost was measured. `SnnProfilesDeclareTheirGaBudgetExplicitly`-style guards exist for
exactly this class of mistake; the fix here is the LSTM/GRU/Transformer analogue.

### Config schema: `evaluation.ga` is now four blocks, not one

```jsonc
"evaluation": {
  "baselines": ["lstm-ae", "gru-ae", "transformer-ae"],
  "snn_architectures": ["dense", "conv1d", "recurrent"],
  "ga": {
    "snn":         { "population_size": 8, "generations": 5, "min_layers", "max_layers",
                      "min_width", "max_width", "voltage_threshold_min/max", "alpha_min/max", ... },
    "lstm":        { "population_size", "generations", "min_hidden", "max_hidden",
                      "min_layers", "max_layers", ... },
    "gru":         { "population_size", "generations", "min_hidden", "max_hidden",
                      "min_layers", "max_layers", ... },
    "transformer": { "population_size", "generations", "min_d_model", "max_d_model",
                      "head_choices", "min_layers", "max_layers", "min_d_ff", "max_d_ff", ... }
  }
}
```

A profile on the OLD flat `"ga": {"population_size": ..., ...}` schema (pre-2026-09-22,
SNN-only) is **rejected at load**, not silently reinterpreted as "only the SNN arm
searches, the baselines stay fixed" — the same discipline `reject_renamed_time_steps`
uses for the `time_steps` rename. `evaluation.ga.lstm`/`.gru` are only consulted when
`"lstm-ae"`/`"gru-ae"` appears in `evaluation.baselines`; `.transformer` likewise for
`"transformer-ae"`; `.snn` only when `snn_architectures` is non-empty — a family absent
from the profile has no arm and its GA block, if present, is simply unread.

### Where the new code lives

| File | Role |
|---|---|
| `lib/include/Meeting01RecurrentGaGenome.hpp` + `.cpp` | `RecurrentGenome` (shared LSTM/GRU shape), repair/random/crossover/mutate, `to_lstm_cfg`/`to_gru_cfg` |
| `lib/include/Meeting01RecurrentGaFitness.hpp` + `.cpp` | `LstmGaIndividual`/`GruGaIndividual` (the distinct-type overload trick), their `evaluate_individual` overloads |
| `lib/include/Meeting01TransformerGaGenome.hpp` + `.cpp` | `TransformerGenome`, `repair_transformer` (the `d_model % n_heads` invariant), `to_transformer_cfg` |
| `lib/include/Meeting01TransformerGaFitness.hpp` + `.cpp` | `TransformerGaIndividual`, its `evaluate_individual` |
| `Meeting01Experiment.cpp`: `finalize_baseline_selection<Model>` | Shared nested-LOSO final retrain/test/checkpoint/manifest tail for LSTM-AE/GRU-AE/Transformer-AE — one template instead of three near-identical copies, mirroring `finalize_snn_selection`'s shape |
| `Meeting01Experiment.cpp`: `run_lstm_ga_search` / `run_gru_ga_search` / `run_transformer_ga_search` | Build that family's bounds from the profile, run its NSGA-II search, retrain+finalize the winner, return a `FamilyWinnerSummary` |
| `Meeting01Experiment.cpp`: `FamilyWinnerSummary` | `{family_token, val_mse, inference_cost}` — what each of the 4 per-family searches hands back to the comparison step |

Tests: `tests/meeting01_recurrent_ga_gtest.cpp` (`RecurrentGenome` legality, `to_lstm_cfg`/
`to_gru_cfg` regression anchors — including the `(window_size * time_steps) / lstm_frame_size`
seq_len formula and the fixed-`latent_dim` rule — checkpoint round-trip for both
`LstmGaIndividual` and `GruGaIndividual`), `tests/meeting01_transformer_ga_gtest.cpp`
(`TransformerGenome` legality including `d_model % n_heads == 0` under repair/crossover/
mutation, the same seq_len regression anchor, checkpoint round-trip for
`TransformerGaIndividual`).

### Three bugs this change's own smoke-verification step found

The approved plan for this change required an end-to-end smoke run (small windows,
minimal population) confirming all 4 families search, retrain, and appear in the
manifests before calling the work done. That step is what caught these — none were
introduced by the multi-family orchestration code itself; all three were pre-existing
and would have silently corrupted a real GridUnesp run.

1. **`seq_len` missing the `time_steps` factor.** `make_lstm_cfg`/`make_gru_cfg`/
   `make_transformer_cfg` computed `arch.seq_len = window_size / lstm_frame_size`, dating
   from before [the missing-time-axis fix](#the-missing-time-axis-found--fixed-2026-09-22-second-pre-gridunesp-audit)
   made the encoder expand every window into a real `(time_steps, window_size)` tensor.
   The correct formula, `(window_size * time_steps) / lstm_frame_size`, was already
   documented in this file's "Paying for the 16×" section — but only the reasoning had
   been updated, not this one call site. For LSTM/GRU this only under-costs the MAC
   estimate (they infer their real sequence length dynamically at forward time). For the
   Transformer it is **loud**: `TransformerAutoencoder` sizes a *fixed* positional-encoding
   buffer from `seq_len` once at construction, so a wrong value throws "Block indices out
   of range" on the very first real forward pass — which is exactly how this was found.

2. **`Trainer`'s validation loop ignored `batch_size`.** `fit_loop_supervised` stacked the
   *entire* validation set into one `stack_time_major` call, rather than chunking it by
   `cfg_.batch_size` the way the training loop already did. For a frame-consuming sequence
   autoencoder this silently concatenates every validation sample into one impossibly long
   pseudo-sequence — 10 validation windows at `seq_len=128` became one `1280`-row
   tensor — which is **loud** for the same reason as bug 1 (the Transformer's fixed
   positional buffer overruns), but would have been **silent** corruption for any model
   that tolerates an arbitrary `T` at inference (the loss would still compute, just over
   the wrong groupings). Fixed by chunking validation the same way training already was;
   existing SNN/time-major consumers are unaffected because their `batch_size` is
   typically ≥ their validation set size, so the new chunked loop degenerates to the old
   single-batch behavior for them.

3. **The SNN GA's own bottleneck was never pinned to `latent_dim`.** `Genome::encoder_widths.back()`
   carried a comment saying "last = latent" and a `latent()` accessor implying the
   invariant was enforced — but `repair_widths` treated the last width as just another
   free draw in `[min_width, max_width]`, and `Genome::latent()` turned out to have zero
   call sites anywhere in the codebase: a documented invariant nobody actually checked.
   `build_snn_decoder` (`AutoencoderBuilders.hpp`, shared core code) always builds its
   first decoder layer expecting exactly `cfg.model.latent_dim` input features, regardless
   of what the genome's encoder actually output — so any genome whose smallest width
   landed away from `latent_dim` crashed with "Linear layer forward: input features (…)
   do not match expected in_features (…)" on its very first real forward pass. This is the
   SNN-side version of the same rule LSTM/GRU/Transformer already followed correctly
   (§"`latent_dim` is fixed, not evolved") — the SNN just never had it enforced in code.
   Fixed by adding `GenomeBounds::latent_dim` and rewriting `repair_widths` to
   unconditionally force `encoder_widths.back() == bounds.latent_dim`, raising the
   effective minimum for every *hidden* width to `latent_dim + 1` so the strictly-decreasing
   sequence can still terminate there.

All three were confirmed fixed by a full rebuild (0 errors/warnings), the full test suite
(3198 → 3220 after the two new test files, 0 failures), and re-running the 4-family smoke
profile end to end: all four families now train, retrain their GA winner, and write both
their own `_model_selection_manifest.json` and the combined `_run<r>_overall_winner_manifest.json`.

### Cost consequence (stated plainly, not buried)

Before this change: 1 GA search (SNN) + 3 fixed-architecture trainings per (dataset, fold,
encoding, run). After: 4 GA searches, one per family — but **not** of comparable size.
The original estimate here said swapping 3 cheap fixed trainings for 3 more searches
"roughly quadruples" total grid wall-clock time, on the assumption that a baseline
evaluation would cost about what an SNN evaluation costs. The real measurement (see
[Equal TIME budget, not equal evaluation count](#equal-time-budget-not-equal-evaluation-count)
above) found LSTM/GRU/Transformer cost 168–272× more **per epoch** than the SNN, not
roughly the same — and the search budget the user ultimately chose for the three
baselines matches the SNN's own shape (`population=8, generations=5`) rather than a
smaller one, so the full grid lands at an estimated **≈ 860 days worst case** (≈ 28.7
months, cited from `_total_runs_breakdown` after the 2026-09-23 dataset-grid widening
from 3 to 4 datasets — see [Three datasets](#three-datasets-multi-dataset-answer-to-one-small-database)),
not a ~4× multiple of the SNN-only design's already-multi-week cost. This is
inherent to the scope change the user asked for (search every family's architecture, not
just the SNN's) combined with how expensive a truly sequential 512-step recurrent/
attention unroll is at batch size 1, compounded by the deliberate choice to give the
baselines the same search power as the SNN rather than a cheaper, smaller one.

## See Also

- [LSTM and BPTT](../Concepts/LSTM-and-BPTT.md) - Theory
- [SNN and Surrogate Gradients](../Concepts/SNN-and-Surrogate-Gradients.md)
- [Autoencoders](../Concepts/Autoencoders.md)
- [Wave Processing](../Core/Wave.md)
- [Training](../Core/Training.md) - Progress bars
- [AutoencoderRunner](../Experiments/AutoencoderRunner.md) - Feedforward autoencoder

## References

[1] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735

[2] A. Graves, "Generating sequences with recurrent neural networks," arXiv preprint arXiv:1308.0850, 2013. [Online]. Available: https://arxiv.org/abs/1308.0850

[3] FSDD Dataset: https://github.com/Jakobovski/Free-Spoken-Digits-Dataset
