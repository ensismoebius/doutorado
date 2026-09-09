# The Experiment Microscope

A research-grade desktop GUI for **inspecting** the `meeting01` and `thesis`
pipelines — not for running them. It lives outside `software/nn/` at
`software/experiment_microscope/` and reaches into this project through a
purpose-built Python binding.

- **Project + full spec**: `software/experiment_microscope/` (`FIXME.md` is the
  56-section specification)
- **The binding**: `software/nn/src/bindings/` → the `nn_microscope` module
- **Related**: [Core/Wavelet](../Core/Wavelet.md), [Core/Paraconsistent](../Core/Paraconsistent.md),
  [Experiments/Meeting01](../Experiments/Meeting01.md), [Experiments/Thesis](../Experiments/Thesis.md)

---

## The problem it solves

The pipelines answer *"what is this number?"* — a CSV row says
`d_penalized = 1.81`, an event log says `val_mse = 0.014`. They do **not**
answer *"why did this number become this number?"*. To see that today you would
add `printf`s, rebuild, re-run a weeks-long job, and eyeball raw `.npy` dumps.

The microscope lets a researcher start from any artifact and walk the
transformation chain in both directions:

```
d_penalized ── d_truth ── G1 / G2 ── α / β ── feature vectors
    ── feature extraction ── wavelet coefficients ── raw signal
```

without touching the experiment code or its results.

---

## Concrete example

`results/thesis/phase00/…hc_daub10_lfcc_c1_eeg_rep0_paraconsistent.csv` contains:

| label | alpha | beta | d_truth | d_penalized |
|---|---|---|---|---|
| handcrafted-lfcc | 0.25807865 | 0.94065476 | 1.69427192 | 1.81068726 |

In the GUI: open **Thesis → Phase 00 → that run → samples (eeg, live)**, pick
sample #3. The signal view shows its six EEG channels; the Wavelet Lab shows the
16 wavelet-packet leaves of channel 0 with per-band energy; clicking a leaf
plots its coefficients. The paraconsistent plane shows every persisted feature
set at its `(G1, G2)` position, `handcrafted-lfcc` among them.

The number the GUI recomputes for that feature set is
**bit-for-bit identical** to the CSV (all 8 decimals) — because it calls
`thesis::rank_feature_sets`, the same function the experiment called, through
the binding.

---

## The structure

```
 software/experiment_microscope/               (PySide6 app; own venv via ./run.sh)
   src/experiment_microscope/
     core/       SelectionState · AppState · TransformationCache · TimelinePlayer · integrity
     data/       ExperimentAdapter ABC → Meeting01Adapter · ThesisAdapter · ParaconsistentGaAdapter
     processing/ _binding.py (locates the .so) · wavelet · meeting01 · thesis · paraconsistent
     views/      explorer · signal_view · wavelet_lab · paraconsistent_plane · provenance_inspector
                         │
                         │  numpy in / numpy out
                         ▼
 software/nn/src/bindings/  →  nn_microscope.{wavelet, meeting01, thesis}
                         │
                         │  links the real static libs
                         ▼
     wavelet · meeting01_lib · autoencoderRunner_lib · thesis_lib · tensor
```

Two data routes, one per adapter:

| | source of truth | how it is read |
|---|---|---|
| **Persisted results** — event JSONL, `*_comparative_metrics.csv`, `*_paraconsistent.csv`, `summary.json`, `pareto.json` | `software/nn/results/` | plain file reads; `meeting01` also imports `scripts/pipeline/meeting01/monitor.py` directly (`EventTailer`, `SessionState`, `_drain`) |
| **Intermediate representations** — windows, encodings, wavelet leaves, handcrafted features, α/β, AE latents/reconstructions | recomputed on demand | `nn_microscope`, calling the experiment's own C++ |

Every value the GUI displays carries an **origin tag** (`MEASURED`,
`COMPUTED`, `PROJECTED`, `DISPLAY_ONLY`, `ESTIMATED`, `MISSING`) so a
recomputed number is never confused with a loaded one, and a missing metric
renders as `—`, never `0`.

---

## The binding: `nn_microscope`

Built only by the `python-bindings` preset (a superset of `max-performance`).
The default build is unaffected — `option(NN_BUILD_PY_BINDINGS OFF)` gates both
the pybind11 vendor step and the `src/bindings/` subdirectory.

```bash
cd software/nn
cmake --preset=python-bindings
cmake --build out/build/python-bindings --target nn_microscope
# → out/build/python-bindings/src/bindings/nn_microscope*.so
```

pybind11 is vendored via `cmake/VendorPybind11.cmake` (FetchContent, pinned
`v3.1.0`, `FIND_PACKAGE_ARGS` so a compatible system install satisfies it
without a clone).

| submodule | key functions | calls |
|---|---|---|
| `nn_microscope.wavelet` | `decompose(signal, wavelet, mode, level)`, `subband_energies` | `wavelets::malat`, `wavelets::extract_subband_energies` |
| `nn_microscope.meeting01` | `build_split`, `encode_sample`, `apply_snn_architecture_transform`, `flatten_time_series`, `to_lstm_frames`, `snn_ae_forward` | `meeting01::` free functions + `ProtocolSpikingAutoencoder` |
| `nn_microscope.thesis` | `load_dataset`, `extract_handcrafted[_features]`, `rank_feature_sets`, `paraconsistent_score` | `thesis::load_dataset` / `extract_features` / `rank_feature_sets` / `score_feature_set` |

`K_CONTRADICTION_PENALTY` (`2 − √2 ≈ 0.5857864376269049`) is exposed as a module
constant, mirrored from `ThesisParaconsistent.hpp`.

### The numpy ↔ Tensor bridge

`src/bindings/tensor_bridge.hpp`. `nn::Tensor`'s xtensor backend stores a 2-D
`(rows, cols)` array in **column-major** order. A naive `memcpy` from
`data_ptr()` therefore *transposes* the data — silently, with plausible-looking
output. The bridge instead copies element-by-element through `at(r, c)`, so the
mapping is correct regardless of layout, and copies (never aliases) so Python
cannot mutate C++ state.

---

## `snn_ae_forward` and the `.npz` model artifacts

To show a window's SNN-AE latent and reconstruction without retraining, the GUI
must reload the trained encoder/decoder. `save_snn_combo_models`
(`Meeting01Experiment.cpp`) now emits, alongside the human-readable
`*_encoder_params.txt` / `*_decoder_params.txt`:

```
<run_tag>_snn_<role>_<dataset>_<encoding>_<arch>_vth<…>_a<…>_fold<f>_run<r>_encoder.npz
                                                                        …_decoder.npz
```

via `NetworkSerializer::saveNetwork` (the SNN AE's `encoder_` / `decoder_` are
plain `nn::Sequential` of `Linear` / `Lif`, which the serializer round-trips
exactly — see [Core/Saver](../Core/Saver.md)). Written only when
`dataset.save_models` is true; the LOSO profile sets it, and
`01_meeting01_run_loso.sh` clears `results/meeting01/models/` on a fresh run.

---

## Confusable pairs

| | |
|---|---|
| **`nn_microscope`** (this binding) | **`monitor.py`** — a stdlib-only event aggregator the `Meeting01Adapter` *imports*; unrelated to the `.so` |
| **`COMPUTED`** — recomputed from measured data via the C++ core | **`PROJECTED`** — a PCA/t-SNE dimensionality reduction; not a measurement |
| **`src/core/paraconsistent/`** — Da Costa logic primitives | **`thesis::score_feature_set`** — the α/β/G1/G2 feature-quality metric the GUI recomputes |
| **`python-bindings` preset** | **`max-performance` preset** — identical flags, but `NN_BUILD_PY_BINDINGS=OFF`; building `nn_microscope` from it will not work |

---

## Failure modes

- **Binding not built** → *loud*. `processing/_binding.py` raises
  `BindingUnavailableError` naming the missing `.so` **and** the exact build
  command. There is no NumPy fallback — a second implementation of the
  scientific pipeline is exactly what would make GUI numbers untraceable
  (FIXME §3, §198; matches this project's no-fallback policy).
- **A metric is absent for the selected object** → *loud*. Rendered `—`. The
  integrity layer forbids `N/A → 0`.
- **Column-major mishandling in a new binding** → *would be silent*. Always
  round-trip a known non-square matrix through `to_numpy` / `from_numpy` when
  adding a submodule (`tests/test_binding_*` do this).
- **Generated caption asserts a conclusion** ("converged", "best") → *loud* in
  tests. `core/integrity.assert_no_inferential_language` mirrors the banned-word
  check `monitor.py` runs on its own dashboard.

---

## Running

```bash
cd software/experiment_microscope
./run.sh                       # bootstraps .venv (--system-site-packages), launches
./.venv/bin/pytest -q          # 27 offscreen tests
./.venv/bin/pytest -q -m slow  # + the bit-exact thesis parity check (~needs ~/database.sqlite)
```

Status (2026-09): foundation + wavelet/paraconsistent/signal views working for
both pipelines; SNN spike/membrane traces, reconstruction and latent explorers,
and the 3D / animation / cross-experiment layers are still to come. Progress is
tracked in the plan file, phases follow `FIXME.md` §52.
