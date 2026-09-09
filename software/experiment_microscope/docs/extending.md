# Extending the microscope

Three kinds of change, smallest first.

## Add a visualization

1. New `QWidget` in `views/`, guarded by `PG_OK` if it uses pyqtgraph:

   ```python
   from experiment_microscope.views._pg import PG_OK, missing_widget, pg

   class MyView(QWidget):
       def __init__(self, repo, selection=None, parent=None):
           super().__init__(parent)
           self.repo = repo
           ...

       def show_node(self, node, adapter_key):
           adapter = self.repo.adapter(adapter_key)
           try:
               payload = adapter.load_features(node)   # or load_signal, ...
           except NotImplementedError:
               self._status.setText("nothing to show for this object")
               return
           except BindingUnavailableError as exc:
               self._status.setText(str(exc))          # shows the build command
               return
           self._render(payload)
   ```

2. Register it in `Workspace._build_central`: construct it, `self.tabs.addTab(view, "My View")`,
   and call `view.show_node(...)` from `Workspace._on_node_selected`.

3. Publication export (optional): add `can_export()` and
   `export_figure(path, **opts)` using `viz/mpl_export.py::new_figure` /
   `save_figure` / `annotate_provenance`. The Export menu picks it up
   automatically for the active tab.

4. Animation (optional): add `set_timeline(player)`, register a frame count with
   `player.set_total_frames(n)` when a selection lands, and connect
   `player.frame_changed` to your per-frame slot. Pull frame data from
   `repo.cache`, never recompute in the slot.

5. Add a test in `tests/` (offscreen `qapp` fixture is session-scoped; use
   `first_fsdd_window` for a real meeting01 window).

## Add a data adapter

Subclass `ExperimentAdapter` (`data/adapters.py`):

```python
class MyAdapter(ExperimentAdapter):
    key = "myexp"
    title = "My Experiment"

    def root_nodes(self): ...
    def children(self, node): ...
    # override only the payloads you can supply:
    def load_signal(self, node) -> Signal1D: ...
    def paraconsistent_points(self) -> list[ParaconsistentPoint]: ...
```

- Return `TreeNode(kind, label, handle, metadata, has_children)`; `handle` is
  opaque and handed back to `children()` / `load_*` to descend.
- Wrap every number in a `Value` with an `Origin`. Unknown → `Value.missing()`
  (renders `—`), never `0` or a guess.
- Anything requiring recompute without the `.so`: `raise
  BindingUnavailableError(...)` (import from `processing._binding`).
- Register it in `DataRepository.__init__`'s adapter tuple.

## Add an experiment (new pipeline)

An "experiment" is just an adapter plus, usually, new binding functions.

1. **Binding** (`software/nn/src/bindings/`): add `bind_myexp.cpp`, call it from
   `module.cpp`, list its source + link its static lib in
   `src/bindings/CMakeLists.txt`. The concrete (non-templated) `myexp::` free
   functions are what you bind — never re-implement them in Python.
   Rebuild: `cmake --build out/build/python-bindings --target nn_microscope`.
2. **Tensor bridge**: `nn::Tensor` is column-major for the 2-D `(rows,cols)`
   case — copy element-by-element via `at(r,c)` (`tensor_bridge.hpp` already
   does this). A raw `memcpy` silently transposes.
3. **Processing wrapper** (`processing/myexp.py`): thin dataclass + function
   layer over the binding, no science.
4. **Adapter** as above.
5. **Parity test**: assert the live chain reproduces a persisted artifact to
   float tolerance (see `tests/test_thesis_parity.py`), marked `@pytest.mark.slow`.
6. **Pipeline DAG**: add the stage graph in `views/pipeline_dag.py::_GRAPHS`.
7. **Follow-Data strip**: extend `views/follow_data.py::_availability` so the
   right stages light up for the new node kinds.

## Conventions

- `pyproject.toml` groups related classes per module (the `efficient_nn_lab`
  template). The `code_intelligence` "one type per file" violations are that
  engine's opinion, not this project's — do not act on them.
- No fallbacks. A missing dependency, artifact or `.so` raises an exception
  naming the cause **and** the remedy.
- Caveman-terse docstrings; didactic (`problem → concrete example → drawn
  structure`) for anything explaining *why*.
