"""Developer / DEBUG panel (FIXME §37).

An opt-in dock that reports the app's own runtime health — frame time, the
transformation cache hit rate, worker-pool occupancy — next to a set of
*scientific consistency checks*: the shape of the signal at each pipeline stage
that currently has a live representation. A shape that is present but does not
line up with its neighbour (e.g. a wavelet coefficient count that is not a power
of two, a reconstruction whose length differs from the raw window) is the loud
failure this panel exists to make visible.

Nothing here recomputes anything: it only reads what the other views already
hold, so opening the panel is free.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.cache import TransformationCache

ShapeProbe = Callable[[], "dict[str, tuple[int, ...] | None]"]


def _fmt_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "—"  # MISSING, never 0 (FIXME §33)
    return " × ".join(str(d) for d in shape)


class DeveloperPanel(QWidget):
    #: pipeline stages reported under "scientific consistency checks"
    STAGES = ("raw", "transformed", "latent", "reconstruction")

    def __init__(
        self,
        cache: TransformationCache,
        shape_probe: ShapeProbe,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache = cache
        self._probe = shape_probe
        self._last_tick = time.perf_counter()
        self._frame_ms = 0.0

        root = QVBoxLayout(self)

        perf = QGroupBox("runtime")
        pf = QFormLayout(perf)
        self._l_frame = QLabel("—")
        self._l_hit = QLabel("—")
        self._l_entries = QLabel("—")
        self._l_workers = QLabel("—")
        self._l_queue = QLabel("—")
        pf.addRow("frame time", self._l_frame)
        pf.addRow("cache hit rate", self._l_hit)
        pf.addRow("cache entries", self._l_entries)
        pf.addRow("active workers", self._l_workers)
        pf.addRow("in-flight jobs", self._l_queue)
        root.addWidget(perf)

        checks = QGroupBox("scientific consistency checks")
        cf = QFormLayout(checks)
        self._l_stage: dict[str, QLabel] = {}
        for stage in self.STAGES:
            lab = QLabel("—")
            self._l_stage[stage] = lab
            cf.addRow(f"{stage} shape", lab)
        self._l_verdict = QLabel("—")
        self._l_verdict.setWordWrap(True)
        cf.addRow("verdict", self._l_verdict)
        root.addWidget(checks)
        root.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)

    # -- lifecycle ------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        self._timer.start()
        self.refresh()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    # -- refresh ------------------------------------------------
    def refresh(self) -> None:
        now = time.perf_counter()
        self._frame_ms = (now - self._last_tick) * 1000.0
        self._last_tick = now

        st = self._cache.stats()
        self._l_frame.setText(f"{self._frame_ms:.0f} ms since last tick")
        self._l_hit.setText(
            f"{st['hit_rate'] * 100:.0f}%  ({st['hits']} hit / {st['misses']} miss)"
        )
        self._l_entries.setText(f"{st['entries']} / {st['max_entries']}")
        self._l_workers.setText(str(st["pool_active"]))
        self._l_queue.setText(str(st["inflight"]))

        try:
            shapes = self._probe() or {}
        except Exception as exc:  # noqa: BLE001
            shapes = {}
            self._l_verdict.setText(f"shape probe failed: {exc}")
        for stage in self.STAGES:
            self._l_stage[stage].setText(_fmt_shape(shapes.get(stage)))
        self._l_verdict.setText(self._consistency_verdict(shapes))

    def _consistency_verdict(self, shapes: "dict[str, tuple[int, ...] | None]") -> str:
        problems: list[str] = []
        raw = shapes.get("raw")
        transformed = shapes.get("transformed")
        recon = shapes.get("reconstruction")
        if transformed is not None:
            n = int(np.prod(transformed))
            if n and (n & (n - 1)) != 0:
                problems.append(f"transformed size {n} is not a power of two")
        if raw is not None and recon is not None:
            if int(np.prod(raw)) != int(np.prod(recon)):
                problems.append(
                    f"reconstruction {_fmt_shape(recon)} does not match raw {_fmt_shape(raw)}"
                )
        if not any(shapes.get(s) is not None for s in self.STAGES):
            return "no live representation — select a sample"
        return "consistent" if not problems else "LOUD: " + "; ".join(problems)
