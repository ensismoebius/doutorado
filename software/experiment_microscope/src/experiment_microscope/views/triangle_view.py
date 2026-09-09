"""The Wavelet | Features | Paraconsistent triangle (FIXME §44).

The application's signature view. For a selected thesis Phase-00 handcrafted
run it puts three panels side by side, all locked to one sample index:

    WAVELET             FEATURES                PARACONSISTENT
    packet leaves +     this sample's           where this run's feature
    band energy         handcrafted vector      set sits on the G1xG2 plane

Scrubbing the sample spinbox moves all three. Everything is recomputed through
``nn_microscope`` (wavelet + handcrafted features) or read from the persisted
``*_paraconsistent.csv`` — no new numbers are invented.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg


class TriangleView(QWidget):
    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._run_node: TreeNode | None = None
        self._adapter_key: str | None = None

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._sample = QSpinBox()
        self._sample.setRange(0, 0)
        self._sample.valueChanged.connect(self._render)
        bar.addWidget(QLabel("sample"))
        bar.addWidget(self._sample)
        self._status = QLabel("Select a thesis Phase-00 handcrafted run.")
        self._status.setWordWrap(True)
        bar.addWidget(self._status, 1)
        root.addLayout(bar)

        if not PG_OK:
            root.addWidget(missing_widget("Triangle view"))
            self._layout = None
            return
        self._layout = pg.GraphicsLayoutWidget()
        root.addWidget(self._layout, 1)

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._layout is None:
            return
        h = getattr(node, "handle", {}) or {}
        if adapter_key != "thesis" or h.get("level") != "run":
            self._run_node = None
            self._status.setText("Triangle view needs a thesis Phase-00 run node.")
            self._layout.clear()
            return
        self._run_node = node
        self._adapter_key = adapter_key
        try:
            self._matrix = self.repo.adapter(adapter_key).load_features(node)
        except (NotImplementedError, BindingUnavailableError) as exc:
            self._run_node = None
            self._status.setText(str(exc) if str(exc) else "no live feature matrix for this run")
            self._layout.clear()
            return
        self._points = [
            p for p in self.repo.adapter(adapter_key).paraconsistent_points()
            if p.facet.get("run_tag") == h.get("run_tag")
        ]
        n = self._matrix.values.shape[0]
        self._sample.blockSignals(True)
        self._sample.setRange(0, max(0, n - 1))
        self._sample.setValue(0)
        self._sample.blockSignals(False)
        self._render()

    def _render(self) -> None:
        if self._layout is None or self._run_node is None:
            return
        self._layout.clear()
        i = self._sample.value()
        m = self._matrix

        # -- FEATURES (centre) : this sample's handcrafted vector -------------
        vec = np.asarray(m.values[i], dtype=float)
        pf = self._layout.addPlot(row=0, col=1, title=f"features — {m.sample_labels[i]}")
        pf.addItem(pg.BarGraphItem(x=np.arange(vec.size), height=vec, width=0.8,
                                   brush=pg.mkBrush(120, 170, 255)))
        pf.showGrid(x=True, y=True, alpha=0.2)

        # -- WAVELET (left) : sample's channel-0 packet energy ---------------
        pw = self._layout.addPlot(row=0, col=0, title="wavelet packet energy (ch0)")
        try:
            from experiment_microscope.processing import wavelet as wl

            sig = self._sample_signal(i)
            spec = self._spec()
            d = wl.decompose(sig, spec.wavelet, "packet", spec.dtwpt_level)
            e = d.subband_energies
            pw.addItem(pg.BarGraphItem(x=np.arange(e.size), height=e, width=0.8,
                                       brush=pg.mkBrush(255, 190, 90)))
        except Exception as exc:  # noqa: BLE001
            pw.addItem(pg.TextItem(f"wavelet unavailable: {exc}", color=(200, 160, 160)))
        pw.showGrid(x=True, y=True, alpha=0.2)

        # -- PARACONSISTENT (right) : the run's feature set on the plane -----
        pp = self._layout.addPlot(row=0, col=2, title="paraconsistent  G1 x G2")
        pp.setXRange(-1.05, 1.05)
        pp.setYRange(-1.05, 1.05)
        pp.addLine(x=0, pen=pg.mkPen((120, 120, 120)))
        pp.addLine(y=0, pen=pg.mkPen((120, 120, 120)))
        for p in self._points:
            if p.g1.is_missing or p.g2.is_missing:
                continue
            pp.plot([float(p.g1.magnitude)], [float(p.g2.magnitude)], pen=None,
                    symbol="o", symbolSize=12, symbolBrush=pg.mkBrush(120, 230, 140))
        self._status.setText(
            f"{m.set_label}  [{m.origin.value}] — sample {i}/{m.values.shape[0] - 1}, "
            f"{vec.size} features; d_penalized = d_truth + (2-sqrt2)|g2|"
        )

    # -- helpers -----------------------------------------------------
    def _spec(self):
        from experiment_microscope.processing.thesis import HandcraftedSpec

        h = self._run_node.handle
        s = self.repo.adapter("thesis")._summary(h["phase"], h["run_tag"])
        hc = s.get("handcrafted") or {}
        return HandcraftedSpec(
            wavelet=hc.get("wavelet", "daub4"),
            scale=hc.get("scale", "lfcc"),
            cepstral=bool(hc.get("cepstral", False)),
            modality=s.get("modality", "eeg"),
            dtwpt_level=int(hc.get("dtwpt_level", 4)),
        )

    def _sample_signal(self, i: int) -> np.ndarray:
        adapter = self.repo.adapter("thesis")
        view = adapter._view(self._spec().modality)
        s = view.sample(i)
        eeg = np.asarray(s["eeg"])
        return eeg[0] if eeg.ndim == 2 else np.asarray(s["audio"]).reshape(-1)
