# Architecture

## The problem this layout solves

Fifteen views need the same signal, wavelet decomposition and feature matrix.
If each view loaded its own copy, a wavelet decomposition would run fifteen
times, the GUI thread would block on `~/database.sqlite`, and two views could
disagree about the same number. So there is exactly one path from a view to
data, and every derived array is computed once.

```
        ExplorerTree ─┐
        SignalView ───┤
        WaveletLab ───┤    node_selected / selection.changed
        Triangle ─────┼────────────────────────────────┐
        …             │                                │
                      ▼                                ▼
              SelectionState (QObject)          DataRepository
              one `changed(str)` signal         ├─ TransformationCache  (LRU 256, QThreadPool)
              coarse→fine fields, cascade        └─ adapters:
                                                     Meeting01Adapter ─┐
                                                     ThesisAdapter ────┼─→ nn_microscope  (pybind11 → C++)
                                                     ParaconsistentGaAdapter
```

- **`app/workspace.py`** — the `QMainWindow`. Builds the tab stack + docks,
  owns the single `SelectionState`, `AppState`, `TimelinePlayer` and
  `DataRepository`, and is the only place that connects views to each other.
  Views never import one another.
- **`core/selection.py::SelectionState`** — the shared cursor. Fields ordered
  coarse→fine (`experiment, dataset, subject, recording, sample, window,
  time_range, wavelet_node, feature, model, encoding, layer, neuron,
  timestep`). Setting a coarse field clears the finer ones; every change emits
  `changed(field)`. `snapshot()` / `restore()` back the bookmarks.
- **`core/state.py::AppState`** — UI toggles that are not pipeline selection
  (`follow_data_mode`, `low_performance_mode`, `display_downsampled`, `theme`).
- **`data/repository.py::DataRepository`** — owns the adapters and the cache.
  `adapter(key)` and `all_paraconsistent_points()` are the whole surface.

## Data flow for one selection

1. The user clicks a node in `ExplorerTree`; it emits
   `node_selected(TreeNode, adapter_key)`.
2. `Workspace._on_node_selected` writes the coarse `SelectionState` fields and
   calls `show_node(node, key)` on every view.
3. Each view asks `repo.adapter(key)` for what it needs
   (`load_signal`, `load_features`, …). The adapter either returns a persisted
   artifact or calls `nn_microscope` to recompute — wrapping the result in a
   `Signal1D` / `FeatureMatrix` / `ParaconsistentPoint` with an `Origin` tag.
4. Anything expensive and reusable goes through `repo.cache` first.

## Adapters (`data/adapters.py::ExperimentAdapter`)

One ABC, one common vocabulary. The GUI does not know a `meeting01` window
from a `thesis` sample.

| Method | Returns | Notes |
|---|---|---|
| `root_nodes()` / `children(node)` | `list[TreeNode]` | lazy tree; `TreeNode.handle` is opaque, passed back to descend |
| `load_signal(node)` | `Signal1D` | `(n,)` or `(channels, n)` + `sample_rate` + `Origin` |
| `load_features(node)` | `FeatureMatrix` | `(samples, features)` + names + class labels |
| `load_provenance(node)` | `ProvenanceRecord` | four dicts: source / processing / model / artifact, each `str → Value` |
| `load_metrics(node)` | `dict[str, Value]` | EER / AUC / … |
| `load_latent(node, **p)` | `LatentTrace` | latent + reconstruction + optional spikes/v_mem |
| `paraconsistent_points()` | `list[ParaconsistentPoint]` | feeds the G1×G2 plane |

A payload that needs recompute and no `.so` raises `BindingUnavailableError`
(message = the build command). A payload that does not apply raises
`NotImplementedError`. Views catch both and show the reason in place of a plot.

### What each adapter reads

- **`Meeting01Adapter`** imports `scripts/pipeline/meeting01/monitor.py`
  directly for session/fold/epoch state (stdlib-only, import-safe). Windows,
  encodings and AE forward passes come from `nn_microscope.meeting01`. Tree:
  `root → dataset → fold → {test|val|train windows, snn-ae/arch/enc combos} →
  window`.
- **`ThesisAdapter`** reads `results/thesis/{phase00,phase01}/*`. Every
  per-sample number is recomputed from `~/database.sqlite` via
  `nn_microscope.thesis` / `.wavelet`. Tree:
  `root → phase → run → {feature-set results, "samples (modality, live)"} →
  sample`. Per-modality `DatasetView` and per-run `FeatureMatrix` are cached.
- **`ParaconsistentGaAdapter`** reads NSGA-II Pareto fronts; the
  `est_latency_ms` column is tagged `ESTIMATED` "UNCALIBRATED".

## Visualization architecture (`views/`)

Every view is a self-contained `QWidget` with the same two-method contract:

```python
def show_node(self, node: TreeNode, adapter_key: str) -> None: ...   # repopulate from a selection
def can_export(self) -> bool  /  def export_figure(self, path, **opts): ...   # publication figure (§30)
```

- **pyqtgraph** for all 2-D interactive plots, behind `views/_pg.py` (`PG_OK`
  guard + `missing_widget()` fallback so an incomplete install degrades to a
  label, not a crash).
- **`views/_timesync.py::TimeCursor`** — a draggable line + region bound to a
  `PlotItem` and `SelectionState`. Dragging in any panel writes
  `selection.timestep` / `time_range`; every other cursor follows.
  `reattach()` survives `PlotItem.clear()`.
- **`views/transport_bar.py` + `core/animation.py::TimelinePlayer`** — one
  shared transport. A temporal view calls `set_timeline(player)` and registers
  its frame count; the player only ever emits a frame *index*, never data.
- **matplotlib** is used only for file export (`viz/mpl_export.py`,
  `Figure` without pyplot, 300 dpi, `.png/.pdf/.svg`), never on screen.
- **`views/developer_panel.py`** (§37, opt-in) — frame time, cache hit rate,
  worker occupancy, and shape-consistency checks (transformed size a power of
  two? reconstruction length == raw length?).

## Caching (`core/cache.py::TransformationCache`)

- Key = `(adapter, sample_id, stage, params_hash)` (`CacheKey.make`).
- `request(key, fn)` returns the cached value now, or schedules `fn` on
  `QThreadPool.globalInstance()` and returns `None`; when it lands, `ready(key,
  value)` fires on the GUI thread. In-flight duplicates are coalesced.
- `compute_blocking(key, fn)` is the synchronous path for tests and headless
  use.
- LRU, 256 entries, process-local, cleared on exit. `stats()` exposes
  hit/miss/entries/in-flight for the developer panel.
- Failures propagate intact: `failed(key, exc, traceback)` — the GUI shows the
  message unchanged (no-fallback policy, §3).

## Scientific provenance & integrity (`core/integrity.py`)

- `Origin`: `MEASURED | COMPUTED | PROJECTED | DISPLAY_ONLY | ESTIMATED |
  MISSING`. Every `Value` shown in an inspector carries one.
- `Value.missing()` renders as `—`; a missing number is never shown as `0`.
- `assert_no_inferential_language(text)` rejects "best / optimal / significant /
  converged / …" in generated captions unless the artifact establishes the
  claim.
- `ProvenanceRecord` is surfaced verbatim from the `session_begin` event
  (`meeting01`) or `summary.json` (`thesis`): seed, git commit, caps, backend,
  wavelet, level, encoding, `v_th`, latent dim.
