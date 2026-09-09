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
        self._timeline = None
        self._syncing = False

        root = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._sample = QSpinBox()
        self._sample.setRange(0, 0)
        self._sample.valueChanged.connect(self._render)
        self._sample.valueChanged.connect(self._on_spin)
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
        self._layout.scene().sigMouseClicked.connect(self._on_scene_click)
        root.addWidget(self._layout, 1)

        # cross-highlight state (FIXME §44: feature bar <-> wavelet band)
        self._pw = None          # wavelet plot
        self._pf = None          # feature plot
        self._n_bands = 0
        self._per_band = 0       # descriptors per band in the feature vector
        self._cepstral_offset = 0
        self._hl_band: int | None = None
        self._hl_line = None
        self._hl_region = None

    def set_timeline(self, player) -> None:
        """Bind the shared transport so Play scrubs the sample index (FIXME §24)."""
        self._timeline = player
        player.frame_changed.connect(self._on_frame)

    def _on_frame(self, frame: int) -> None:
        if self._run_node is None or self._syncing:
            return
        self._syncing = True
        try:
            self._sample.setValue(max(0, min(self._sample.maximum(), int(frame))))
        finally:
            self._syncing = False

    def _on_spin(self, value: int) -> None:
        if self._timeline is not None and not self._syncing:
            self._syncing = True
            try:
                self._timeline.seek(int(value))
            finally:
                self._syncing = False

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
        if self._timeline is not None:
            self._timeline.set_total_frames(max(1, n))
            self._timeline.seek(0)
        self._render()

    def _render(self) -> None:
        if self._layout is None or self._run_node is None:
            return
        self._layout.clear()
        self._hl_line = None
        self._hl_region = None
        i = self._sample.value()
        m = self._matrix

        # -- FEATURES (centre) : this sample's handcrafted vector -------------
        vec = np.asarray(m.values[i], dtype=float)
        pf = self._layout.addPlot(row=0, col=1, title=f"features — {m.sample_labels[i]}")
        pf.addItem(pg.BarGraphItem(x=np.arange(vec.size), height=vec, width=0.8,
                                   brush=pg.mkBrush(120, 170, 255)))
        pf.showGrid(x=True, y=True, alpha=0.2)
        self._pf = pf

        # -- WAVELET (left) : sample's channel-0 packet energy ---------------
        pw = self._layout.addPlot(row=0, col=0, title="wavelet packet energy (ch0)")
        self._pw = pw
        try:
            from experiment_microscope.processing import wavelet as wl

            sig = self._sample_signal(i)
            spec = self._spec()
            d = wl.decompose(sig, spec.wavelet, "packet", spec.dtwpt_level)
            e = d.subband_energies
            pw.addItem(pg.BarGraphItem(x=np.arange(e.size), height=e, width=0.8,
                                       brush=pg.mkBrush(255, 190, 90)))
            self._map_features_to_bands(spec, int(e.size), int(vec.size))
        except Exception as exc:  # noqa: BLE001
            pw.addItem(pg.TextItem(f"wavelet unavailable: {exc}", color=(200, 160, 160)))
            self._n_bands = 0
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
            + ("  ·  click a bar to link band ⇄ features" if self._n_bands > 0 else "")
        )
        self._apply_highlight()

    # -- cross-highlight (FIXME §44) -------------------------------
    def _map_features_to_bands(self, spec, n_energy_bands: int, n_features: int) -> None:
        """Work out the feature-vector layout so a feature bar maps to a band.

        ``extract_handcrafted`` (ThesisHandcraftedFeatures.cpp) emits, per scale
        group in order: energy?, zcr?, entropy?, teager?, jitter?, shimmer? — the
        wanted descriptors only. In cepstral (``c2``) mode the per-band energy is
        replaced by ``n_bands`` global DCT coefficients written first, so those
        cannot be attributed to one band.
        """
        n_desc = len(spec.descriptors)
        offset = n_energy_bands if spec.cepstral else 0
        per_band = n_desc - (1 if spec.cepstral else 0)  # energy dropped in cepstral mode
        expected = offset + per_band * n_energy_bands
        if per_band > 0 and expected == n_features:
            self._n_bands = n_energy_bands
            self._per_band = per_band
            self._cepstral_offset = offset
        else:  # scale grouping merged leaves — layout not 1:1, disable highlight
            self._n_bands = 0

    def _feature_span_for_band(self, band: int) -> tuple[int, int]:
        lo = self._cepstral_offset + band * self._per_band
        return lo, lo + self._per_band

    def _band_for_feature(self, j: int) -> int | None:
        if j < self._cepstral_offset:
            return None  # a global cepstral coefficient — not one band
        return (j - self._cepstral_offset) // self._per_band

    def _on_scene_click(self, event) -> None:
        if self._n_bands <= 0 or self._pw is None or self._pf is None:
            return
        pos = event.scenePos()
        for plot, to_band in ((self._pw, lambda x: int(round(x))),
                              (self._pf, self._band_for_feature_x)):
            if plot.sceneBoundingRect().contains(pos):
                x = plot.vb.mapSceneToView(pos).x()
                band = to_band(x)
                if band is not None and 0 <= band < self._n_bands:
                    self._hl_band = band
                    self._apply_highlight()
                return

    def _band_for_feature_x(self, x: float) -> int | None:
        return self._band_for_feature(int(round(x)))

    def _apply_highlight(self) -> None:
        if self._pw is None or self._pf is None:
            return
        if self._hl_line is not None:
            self._pw.removeItem(self._hl_line)
            self._hl_line = None
        if self._hl_region is not None:
            self._pf.removeItem(self._hl_region)
            self._hl_region = None
        if self._n_bands <= 0 or self._hl_band is None or self._hl_band >= self._n_bands:
            return
        pen = pg.mkPen((120, 230, 140), width=2)
        self._hl_line = pg.InfiniteLine(pos=self._hl_band, angle=90, pen=pen, movable=False)
        self._pw.addItem(self._hl_line)
        lo, hi = self._feature_span_for_band(self._hl_band)
        self._hl_region = pg.LinearRegionItem(
            values=(lo - 0.5, hi - 0.5), movable=False,
            brush=pg.mkBrush(120, 230, 140, 60),
        )
        self._pf.addItem(self._hl_region)
        self._status.setText(
            f"{self._matrix.set_label} — band {self._hl_band} ⇄ features {lo}–{hi - 1} "
            f"({', '.join(self._spec().descriptors[-self._per_band:])})"
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
