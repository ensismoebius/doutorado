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

Optional flags pre-select what opens:

```bash
./run.sh --experiment thesis          # pre-select a pipeline
./run.sh --tour meeting01              # open straight into the guided tour
./run.sh --tab "Wavelet Lab"          # raise a named tab on start
```

### Presenting to a room

**F5** (`View → Presentation mode`, or `Esc` to leave) makes the window a
full-screen lecture surface: menus, docks, transport and colour key hidden, the
current view's explanation open, larger type. A presenter clicker works —
`Space` / `→` / `PageDown` advance the guided tour (or step the animation when no
tour is running), `←` / `PageUp` go back.

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

- **Guided tour** (`Help → Start guided tour`, `Ctrl+G`) — a left dock that walks
  an audience through one pipeline a step at a time: a plain-language paragraph
  (no undefined jargon), the one number that matters, and Back / Next. Each step
  drives the app — selects the right data, opens the right tab, runs the forward
  pass — so the presenter only talks and clicks Next. "Free explore" hands the
  app back (Segel & Heer's "martini glass": guided first, open after). Stories:
  `core/story.py`; panel: `views/story_panel.py`.
- **One colour language** (`core/palette.py`) — blue = the original signal,
  orange = the model's rebuild, grey = the error, amber = a spike, green =
  membrane charge, red = the firing line, violet = a frequency band, teal = a
  handcrafted measurement, yellow = your current selection. A "Colour key" strip
  under the tab bar shows only the colours the active tab uses.
- **Prescriptive plot titles** (`core/verdict.py`) — the title states the
  takeaway computed from the data ("Near-perfect rebuild — the latent numbers
  kept almost everything, R² 0.99" / "Conflicting evidence — supported *and*
  denied"), not the category. Still passes the banned-inferential-word guard.
- **Português (Brasil)** — `Exibir → Idioma`. A runtime string catalog
  (`core/i18n.py` + `core/i18n_pt_br.py`, `t()` lookup, English fallback for any
  missing key) covers the menus, dock titles, tab labels, the guided tour (both
  pipelines), the verdict sentences, the colour key, the full glossary (~90
  terms), every "How to read this" box, the Follow-the-Data bar, and the control
  labels / status lines of the Signal, Wavelet Lab, Encoding Lab, SNN Lab,
  Feature Matrix, Experiment Timeline, Inspector, Raw-artifact, Reproduce,
  Bookmarks and Search panels. The Portuguese is written to read naturally
  (idiomatic phrasing, not word-for-word) since the tour and help boxes are the
  teaching surface. A few data-derived strings (adapter signal labels, the
  meeting01 session log, `source:` plot captions) are still English. The choice
  persists in `QSettings`; menus, tabs, the tour and the glossary re-render live,
  other fixed labels update on the next node selection or launch.
- **Scale-to-fit + motion** — every line plot re-fits to its data on each
  render (`_plotinfo.autofit`) and fades in (`_plotinfo.fade_in`); the guided
  tour steps cross-fade.
- **Busy indicator** — a moving bar + plain label in the status bar whenever the
  app is loading a dataset, running the network, or projecting the latent space
  (`views/_busy.py`, fed by `TransformationCache.busy_changed` and the latent
  worker thread).

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
- **Every data plot has a grey `source:` caption** (`_plotinfo.py::set_source`)
  naming the adapter call / file / origin tag it was drawn from.
- **Hover a line plot** and a label follows the cursor with the x position and
  every curve's y value there, plus a vertical guide (`_plotinfo.py::HoverReadout`).
  Scatter plots (paraconsistent plane / landscape, NSGA-II, Latent Space) show a
  per-point tooltip on hover instead.
- **Tabs are selection-aware** (`workspace.py::_update_tab_visibility`): a
  meeting01 window hides the thesis-only tabs (Feature Matrix, Triangle, NSGA-II,
  the paraconsistent scatters); a thesis run hides the meeting01-only tabs
  (SNN Lab / 3D, Encoding Lab, Reconstruction, Latent Space, Timeline). Comparison,
  Ranking and Pipeline are always shown.
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
- **Latent Space Explorer** (§19, §45) — the "Latent Space" tab. A checklist lets
  you tick **one or more meeting01 folds**; every test/val/train window of each
  checked fold runs through its own trained SNN-AE (worker thread, capped), and
  all their latent vectors are projected together with PCA or t-SNE (2-D
  scatter, or 3-D PCA in the VTK panel). Colour by digit, speaker, or — once 2+
  folds are checked — **fold**, to see whether the folds agree or one model
  landed somewhere different. A fold with no trained checkpoint is skipped and
  named in the status line, never silently replaced by another fold's model.
  Every projection is tagged `PROJECTED`. Clicking a point selects that window
  across the whole app (raw signal, wavelet, reconstruction, …).
- **Autoencoder graph** (§18, §19) — the "Autoencoder" tab draws the **whole**
  trained network, encoder *and* decoder, as one left-to-right graph:
  `input(256) → 64 → latent(32) → 64 → reconstruction(256)`. Every dot is a
  neuron (colour/size = its activation for the window); lines are the `Linear`
  weights (blue `+`, red `−`, top-K per neuron). Pressing ▶ floods the signal
  column by column, in to the latent code and back out. Zooming in on a spiking
  column — or the *Neuron detail* toggle — replaces its dots with per-neuron
  gauges: built-up membrane charge, dashed firing line, `⚡` if it fired. Click
  any neuron for its exact numbers. Both halves come from one
  `nn_microscope.meeting01.snn_ae_forward` call (`encoder_layers` +
  `decoder_layers`). A **model picker + "Run this model"** button lets you run
  any trained model for the window's fold explicitly (`Meeting01Adapter.
  models_for` / `load_ae_trace(..., spec_override=...)`) instead of always
  taking the auto-matched winner — useful for comparing two runs/seeds/
  architectures on the same window.
- **Model Structure** (§15, §18, §19) — the "Model Structure" tab shows the
  selected model's topology on its own: every layer, its shape, its trainable
  weight count, and each spiking layer's trained threshold — a property of the
  *checkpoint*, unlike the Autoencoder tab's per-window activations. It follows
  whichever model is currently loaded there (`AutoencoderView.trace_changed`),
  automatically on ordinary browsing and immediately after "Run this model".
- **3D views**: "Wavelet 3D" (§11), "SNN 3D" (§18 — encoder-only neuron columns
  `input → Linear(64) → LIF(64) → latent(32)`, node size/colour = activity,
  edges = top-K `|Linear weight|` per target neuron), and the Latent Space
  Explorer's 3-D PCA panel. Pressing ▶ floods the signal layer-by-layer through
  the net (§17); the shared `TimelinePlayer` drives the flood frames. **Every 3D
  view always draws a visible coordinate box** (`show_grid` + `show_axes`) whose
  ticks are real values, never a cosmetic scale factor baked into the mesh —
  Wavelet 3D's axes are the true coefficient index / leaf number / |coeff|-max
  ratio, SNN 3D's Y axis is explicitly titled "neuron slot (layout order)"
  rather than pretending a layout position is a measurement, and the Latent
  Explorer's 3-D axes are titled "arbitrary — a direction, not a measurement"
  (PCA components have no physical unit by construction).
- **`Meeting01Adapter` caches `load_ae_trace`** per (window, model) and
  `_snn_model_specs()` for a few seconds — opening a window used to re-run the
  same SNN-AE forward pass up to 4 times (SNN Lab, SNN 3D, Autoencoder,
  Reconstruction) and re-glob `results/meeting01/models/**` up to 3 times; both
  are now shared across the views that ask for the same thing in one click.
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
  run writes `results/meeting01/*_events.jsonl`. **Comparing folds**: Ctrl/Shift-click
  several configs (or a whole fold row, which stands in for every config under
  it) to overlay their curves — one colour per config, solid = validation loss,
  dotted = train loss — to see whether the network behaves the same way across
  folds or diverges on one.
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
