# Experiment Microscope — documentation

A desktop tool for answering one question about the `meeting01` and `thesis`
pipelines: **"why did this number become this number?"** — not just *what* a
value is, but which raw sample, window, wavelet band, feature and score it came
from, and what each step did to it.

| Topic | Page |
|---|---|
| How it is put together, how data flows, caching, provenance | [architecture.md](architecture.md) |
| Adding an experiment, an adapter, or a visualization | [extending.md](extending.md) |
| What it cannot do yet | this page, "Limitations" |

---

## Starting the GUI

```bash
cd software/experiment_microscope
./run.sh                       # bootstraps .venv/ (--system-site-packages) and launches
```

Headless / CI:

```bash
QT_QPA_PLATFORM=offscreen ./.venv/bin/python -m experiment_microscope
./.venv/bin/pytest -q          # 63 fast tests; `-m slow` adds the bit-exact parity checks
```

Optional first argument pre-selects the pipeline:

```bash
./run.sh --experiment thesis
```

### The C++ binding

Every recomputed intermediate goes through `nn_microscope`, a pybind11 module
that links the **actual** `meeting01` / `thesis` C++ libraries. There is no
second Python implementation of the science (that is the one thing `FIXME.md`
§3 forbids). Build it once:

```bash
cd software/nn
cmake --preset=python-bindings
cmake --build out/build/python-bindings --target nn_microscope
```

Without the `.so` the app still opens and browses persisted CSV/JSON artifacts,
but any view that needs a recomputed signal, wavelet, feature matrix or score
raises with exactly that build command — it never shows a plausible-looking
fallback number (loud failure, §33).

---

## Supported experiments

| Key | Pipeline | Entered from | Recomputed live via |
|---|---|---|---|
| `meeting01` | nested-LOSO SNN vs LSTM/GRU/Transformer autoencoders (FSDD / AudioMNIST / MIT-BIH) | `results/meeting01/*_events.jsonl`, `*_comparative_metrics.csv`; `~/…/free-spoken-digit-dataset` corpus | `nn_microscope.meeting01` (`build_split`, `encode_sample`, `apply_snn_architecture_transform`, `snn_ae_forward`) |
| `thesis` | handcrafted wavelet + paraconsistent feature selection + DSNN authentication | `results/thesis/phase00,phase01/*_{paraconsistent.csv,summary.json,metrics.csv}`; `~/database.sqlite` | `nn_microscope.thesis` + `nn_microscope.wavelet` (`load_dataset`, `extract_handcrafted_features`, `rank_feature_sets`) |
| `paraconsistent_ga` | NSGA-II architecture search | `results/paraconsistentGA/pga_*_{pareto.json,individuals.csv}` | — (reads persisted Pareto fronts only; "NSGA-II" tab) |

`meeting01` was purged of persisted per-run results and works purely by live
recompute; `thesis` never persisted anything per-sample, so it always did.

---

## Scientific provenance in one example

Select a `thesis` Phase-00 handcrafted run, open the **Triangle** tab, scrub to
one sample. The three panels are the same number at three stages:

```
raw EEG (6×4096 @ 1024 Hz)                         [MEASURED]
    │  nn_microscope.wavelet.decompose(ch0, daub10, packet, level 4)
    ▼
16 packet sub-band energies                        [COMPUTED]
    │  nn_microscope.thesis.extract_handcrafted (energy, zcr, entropy, teager, jitter, shimmer)
    ▼
96-D handcrafted feature vector                    [COMPUTED]
    │  nn_microscope.thesis.rank_feature_sets  (group by subject_id)
    ▼
α, β  →  g1 = α−β,  g2 = α+β−1
        d_truth = ‖(g1,g2) − (1,0)‖
        d_penalized = d_truth + (2−√2)·|g2|        [COMPUTED]
```

The parity test `tests/test_thesis_parity.py` asserts this live chain
reproduces the persisted `*_paraconsistent.csv` row to 8 decimals
(`α=0.25807865, β=0.94065476, d_penalized=1.81068726` for
`hc_daub10_lfcc_c1_eeg`).

Every value shown in an inspector carries an **origin tag**
(`core/integrity.py::Origin`): `MEASURED`, `COMPUTED`, `PROJECTED`,
`DISPLAY_ONLY`, `ESTIMATED`, `MISSING`. `MISSING` renders as `—`, never `0`.
Generated captions are checked against a banned-inferential-word list
(`assert_no_inferential_language`) so the tool never writes "best" / "optimal" /
"significant" about something the artifact does not establish.

## Didactic layer (FIXME §29, §55)

Nothing in the app is shown without an explanation:

- **Every central tab has a "How to read this" strip** (`views/_help.py::HelpBox`)
  — collapsed by default, one click to open. It says what the view shows, what
  each axis and number means (abbreviations spelled out), and what to do next.
  Press **F1** (`Help → Explain the current view`) to open it.
- **`Help → Glossary`** lists every term — MSE, MAE, R², SNN, LIF, LOSO, PCA,
  t-SNE, v_th, α/β, G1/G2, D_truth, D_penalized, ZCR, LFCC, EER, AUC, … — each
  with its expansion and a one-sentence plain-language meaning. The whole dialog
  is generated from `core/glossary.py`, which is also the single source the views
  pull their wording from (`glossary.expand` / `describe` / `tooltip`).
- **Metric tables are three columns**: quantity · value · *what it means*
  (`_help.py::metric_table`). No bare number.
- **Plots carry axis labels with units and a legend** (`_help.py::label_plot`);
  status lines spell out the origin tag and units in words.
- **🔊 Listen** (Signal tab) plays the shown waveform through the default audio
  output at its own sample rate (`viz/audio.py::AudioPlayer`, `QtMultimedia`
  `QAudioSink`; amplitude peak-normalised for listening only). Audio samples
  only — EEG and sub-3 kHz signals disable the button. The **▶** transport is
  unrelated: it steps animation frames, not sound.

---

## Limitations

- **Membrane potentials — two views, two time axes.** meeting01's `recurrent`
  transform sweeps the 256 window samples as time steps
  (`v[t] = α·v[t-1] + x[t] - s[t-1]·v_th`); `nn_microscope.meeting01.recurrent_lif_trace`
  now returns that full `v[t]` trajectory alongside the spike train, and the
  "SNN Lab" tab's third panel plots it (threshold line + spike markers, x-linked
  to the input, animatable). Separately, the SNN-**AE** runs with
  `time_steps == 1` — the window is a feature vector, not a sequence — so
  `snn_ae_forward`'s `encoder_layers` gives one `v_mem` value per LIF neuron; the
  SNN Lab shows that as a fourth *snapshot* panel when a trained `.npz` exists.
  Clicking any spike marker (input raster or membrane panel) prints its exact
  time / membrane / threshold (§16).
- **The Reconstruction tab is live once a LOSO fold has written Step E `.npz`
  models** (`results/meeting01/models/**/…_encoder.npz` / `_decoder.npz`).
  `Meeting01Adapter.load_latent` picks the matching model (prefers the retrained
  `final`), runs `nn_microscope.meeting01.snn_ae_forward`, and the view shows
  original vs reconstruction + residual with MSE / MAE / R² / Pearson r. With no
  model on disk it shows the exact `01_meeting01_run_loso.sh` command instead.
  Checkpoints written before the `NetworkSerializer` `LifBPTT` fix (2026-09-09)
  load their `Linear` weights but not the LIF `R`/`C`/`v_th`; the adapter detects
  this (`_npz_has_lif_params`) and the trace carries an explicit `lif_params`
  caveat (`Origin.ESTIMATED`) shown in the metrics table — re-running the fold
  with the rebuilt binary produces complete checkpoints.
- **Latent Space Explorer** (§19) — the "Latent Space" tab. Runs every
  test/val/train window of the selected meeting01 fold through the trained
  SNN-AE (worker thread, capped), projects the latent vectors with PCA or t-SNE
  (2-D scatter, or 3-D PCA in the VTK panel), colours by digit or speaker.
  Every projection is tagged `PROJECTED`. Clicking a point selects that window
  across the whole app (raw signal, wavelet, reconstruction, …).
- **3D views**: "Wavelet 3D" (§11) and "SNN 3D" (§18 — encoder neuron columns
  `input → Linear(64) → LIF(64) → latent(32)`, node size/colour = activity,
  edges = top-K `|Linear weight|` per target neuron). Pressing ▶ floods the
  signal layer-by-layer through the net (§17); the shared `TimelinePlayer`
  drives the flood frames.
- **Triangle feature-bar ↔ wavelet-leaf cross-highlight** is wired only when the
  handcrafted layout is 1:1 with the wavelet bands (else disabled, no guess) —
  the C++ does not expose a per-feature→band map.
- **`meeting01` animation** covers the recurrent-LIF membrane trajectory (SNN
  Lab) and the SNN-3D layer flood; the Triangle sample index is driven for
  thesis runs.
- **Cache is process-local and unbounded in time** — it is an LRU of 256
  derived representations, cleared on exit, not persisted between sessions.
- The G1×G2 plane (§12) is the "Paraconsistent plane" tab; the D_truth × D_penalized
feature landscape with facet filters (§13) is the "Paraconsistent landscape" tab.
Cross-experiment comparison (§22) is the "Comparison" tab; the cross-run
  ranking / model-comparison table (§21/§12) is the "Ranking" tab; the meeting01
  session→fold→config→epoch timeline (§45) is the "Timeline" tab, with the
  selected config's train/val loss curve below it — with epoch-duration on a
  linked right axis and lr / s-per-epoch in the title (§46); empty until a LOSO
  run writes `results/meeting01/*_events.jsonl`.
- **Themes** (§34): View → Theme → System / Light / Dark, persisted in
  `QSettings`. Colour is never the only signal — line style, markers and labels
  carry the same information.
- **Display resolution** (§26): signals longer than 20 000 samples/channel are
  decimated for the plot and the status bar shows `DISPLAY-DOWNSAMPLED 1:N`; the
  cursor still reads the full-resolution array.
- **Low-performance mode** (§38): View → Low-performance mode disables the 3D
  panels and the animation transport.
- **Global search** (§35): the box above the Data Explorer tree — every
  whitespace token must match a node's path + metadata (e.g. `daub10 lfcc eeg`,
  `fold 0 fsdd`, `pga snn`); activating a hit navigates the tree. The catalog is
  built on first use (bounded-depth force-expand).
