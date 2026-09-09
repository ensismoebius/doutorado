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
| `paraconsistent_ga` | NSGA-II feature-set search | `results/paraconsistentGA/pga_*_{pareto.json,individuals.csv}` | — (reads persisted Pareto fronts only) |

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

---

## Limitations

- **No reconstruction or latent-trajectory view yet.** Both need a trained SNN
  autoencoder `.npz`; `meeting01`'s were purged and the reprocess is a
  multi-hour guarded run. `snn_ae_forward` works but `meeting01`'s `time_steps=1`
  gives a single-step forward pass, so there is no spike raster over time.
- **3D is only the wavelet coefficient landscape** (§11) — the "Wavelet 3D" tab
  (X = coeff index, Y = packet leaf, Z = magnitude, threshold + isolate-leaf
  controls). The SNN 3D activity view (§18) and latent explorer (§19) are not
  built.
- **Triangle feature-bar ↔ wavelet-leaf cross-highlight** is not wired: the C++
  does not expose which wavelet band each handcrafted feature came from.
- **`meeting01` animation** is limited to what `build_split` returns; the
  transport currently drives only the Triangle sample index.
- **Cache is process-local and unbounded in time** — it is an LRU of 256
  derived representations, cleared on exit, not persisted between sessions.
- Cross-experiment comparison (§22) is the "Comparison" tab; the cross-run
  ranking / model-comparison table (§21/§12) is the "Ranking" tab; the meeting01
  session→fold→config→epoch timeline (§45) is the "Timeline" tab (empty until a
  LOSO run writes `results/meeting01/*_events.jsonl`).
