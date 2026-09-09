"""The single shared selection state (FIXME §42).

Every view reads and writes the *same* ``SelectionState`` and reacts to its
``changed`` signal. Views never call each other directly — that is what keeps
the application a coherent instrument instead of a pile of coupled widgets
(FIXME §0, §19: "The GUI should never treat plots as isolated pictures").

The field set matches FIXME §42 literally. Setting a field to the value it
already holds is a no-op and emits nothing (same discipline as
``efficient_nn_lab/core/state.py``).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Signal

#: Ordered coarse-to-fine, matching FIXME §26's multi-resolution hierarchy.
#: Changing a coarse field invalidates every finer one.
FIELDS: tuple[str, ...] = (
    "experiment",   # "meeting01" | "thesis" | "paraconsistent_ga"
    "dataset",      # e.g. "fsdd", "eeg", "voice"
    "subject",      # speaker id / EEG subject id
    "recording",    # recording / trial id
    "sample",       # opaque per-adapter sample handle
    "window",       # window index within the sample
    "time_range",   # (t0, t1) in samples, or None
    "wavelet_node", # (level, path) tuple, or None
    "feature",      # feature index / name within the selected feature set
    "model",        # e.g. "snn-ae/dense", "lstm-ae"
    "encoding",     # "direct" | "poisson" | "latency"
    "layer",        # layer index within the model
    "neuron",       # neuron index within the layer
    "timestep",     # time step within the encoded window
)

_FINER_THAN: dict[str, tuple[str, ...]] = {
    field: FIELDS[i + 1 :] for i, field in enumerate(FIELDS)
}


class SelectionState(QObject):
    """Cross-cutting pointer into the pipeline. One instance per window."""

    #: Emitted once per accepted change; argument is the field name that
    #: changed. A coarse change emits only for that field, but also clears
    #: the finer fields (each of which emits in turn).
    changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._values: dict[str, Any] = {f: None for f in FIELDS}

    def get(self, field: str) -> Any:
        return self._values[field]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        if name in FIELDS:
            return self._values[name]
        raise AttributeError(name)

    def set(self, field: str, value: Any, *, cascade: bool = True) -> None:
        if field not in FIELDS:
            raise KeyError(f"unknown selection field {field!r}; valid: {FIELDS}")
        if self._values[field] == value:
            return
        self._values[field] = value
        if cascade and value is not None:
            for finer in _FINER_THAN[field]:
                if self._values[finer] is not None:
                    self._values[finer] = None
                    self.changed.emit(finer)
        self.changed.emit(field)

    def update(self, **fields: Any) -> None:
        """Set several fields; cascade runs once, after all are applied."""
        touched = []
        for field, value in fields.items():
            if field not in FIELDS:
                raise KeyError(f"unknown selection field {field!r}")
            if self._values[field] != value:
                self._values[field] = value
                touched.append(field)
        if not touched:
            return
        coarsest = min(touched, key=FIELDS.index)
        for finer in _FINER_THAN[coarsest]:
            if finer not in fields and self._values[finer] is not None:
                self._values[finer] = None
                touched.append(finer)
        for field in touched:
            self.changed.emit(field)

    def snapshot(self) -> dict[str, Any]:
        """A plain dict copy — used by bookmarks (FIXME §36)."""
        return dict(self._values)

    def restore(self, snapshot: dict[str, Any]) -> None:
        for field in FIELDS:
            self._values[field] = snapshot.get(field)
        self.changed.emit("*")
