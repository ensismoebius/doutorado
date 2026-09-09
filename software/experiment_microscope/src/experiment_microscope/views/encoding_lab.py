"""Encoding Lab (FIXME §16, §17, Journey B).

For a meeting01 window: show the normalized signal and the spike train each
encoding produces from it, so the researcher can watch

    normalized value -> spike timing

for ``direct`` / ``poisson`` / ``latency`` and compare them. Everything is
computed by ``nn_microscope.meeting01.encode_sample`` — the exact transform the
experiment applies before the SNN autoencoder. No trained model needed.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.views._help import HelpBox
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.core.selection import SelectionState
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.views._timesync import TimeCursor

_ENCODINGS = ("direct", "poisson", "latency")


_HELP = """
<b>What this shows.</b> The three spike <b>encodings</b> side by side for one
window, so you can see how each turns a real number into spike events.
<br><br>
<b>direct</b> — the normalised sample passes straight through (no spike-time
conversion). <b>poisson</b> — a random spike train whose average <b>firing
rate</b> is proportional to the value. <b>latency</b> — time-to-first-spike:
larger values fire earlier, small values late or never.
<br><br>
Horizontal axis = time step within the window; each row/track is a spike train,
each mark a spike. <i>seed</i> fixes the randomness of the poisson encoding so the
picture is reproducible.
"""



class EncodingLab(QWidget):
    def __init__(
        self,
        repo: DataRepository,
        selection: SelectionState | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self._selection = selection
        self._cursor: TimeCursor | None = None
        self._window: np.ndarray | None = None

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Encoding Lab', _HELP))
        top = QHBoxLayout()
        self._seed = QSpinBox()
        self._seed.setRange(0, 2**31 - 1)
        self._seed.setValue(0)
        self._seed.valueChanged.connect(self._render)
        self._mode = QComboBox()
        self._mode.addItems(["compare all", *(_ENCODINGS)])
        self._mode.currentIndexChanged.connect(self._render)
        top.addWidget(QLabel("seed"))
        top.addWidget(self._seed)
        top.addWidget(QLabel("show"))
        top.addWidget(self._mode)
        self._status = QLabel("Select a meeting01 window.")
        self._status.setWordWrap(True)
        top.addWidget(self._status, 1)
        root.addLayout(top)

        if not PG_OK:
            root.addWidget(missing_widget("Encoding Lab"))
            self._layout_widget = None
            return
        self._layout_widget = pg.GraphicsLayoutWidget()
        root.addWidget(self._layout_widget, 1)

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._layout_widget is None:
            return
        if adapter_key != "meeting01" or (getattr(node, "handle", {}) or {}).get("level") != "window":
            self._window = None
            self._status.setText("Encoding Lab needs a meeting01 window.")
            self._layout_widget.clear()
            return
        adapter = self.repo.adapter(adapter_key)
        try:
            sig: Signal1D = adapter.load_signal(node)
        except BindingUnavailableError as exc:
            self._window = None
            self._status.setText(str(exc))
            return
        self._window = np.asarray(sig.samples, dtype=float).reshape(-1)
        self._label = sig.label
        self._render()

    def _encode(self, encoding: str) -> np.ndarray:
        from experiment_microscope.processing import meeting01 as m

        col = self._window.reshape(-1, 1)
        return np.asarray(m.encode(col, encoding, self._seed.value())).reshape(-1)

    def _render(self) -> None:
        if self._layout_widget is None or self._window is None:
            return
        self._layout_widget.clear()
        t = np.arange(self._window.size)

        p0 = self._layout_widget.addPlot(row=0, col=0)
        p0.setTitle("normalized window (z-score)")
        p0.plot(t, self._window, pen=pg.mkPen((120, 170, 255)))
        p0.showGrid(x=True, y=True, alpha=0.2)
        if self._selection is not None:
            if self._cursor is None:
                self._cursor = TimeCursor(p0, self._selection)
            else:
                self._cursor.rebind(p0)  # plots are rebuilt every render

        which = self._mode.currentText()
        encs = _ENCODINGS if which == "compare all" else (which,)
        try:
            for i, enc in enumerate(encs, start=1):
                data = self._encode(enc)
                p = self._layout_widget.addPlot(row=i, col=0)
                p.setXLink(p0)
                p.showGrid(x=True, y=True, alpha=0.2)
                if enc == "direct":
                    p.setTitle("direct (identity)")
                    p.plot(t, data, pen=pg.mkPen((160, 160, 160)))
                else:
                    spikes = np.flatnonzero(data > 0.5)
                    p.setTitle(f"{enc} — {spikes.size} spike(s)")
                    p.plot(
                        spikes, np.ones_like(spikes, dtype=float),
                        pen=None, symbol="|", symbolSize=12,
                        symbolPen=pg.mkPen((255, 190, 90)),
                    )
                    p.setYRange(0.0, 1.5)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"encode failed: {exc}")
            return
        self._status.setText(f"{getattr(self, '_label', 'window')}  [computed] — seed {self._seed.value()}")
