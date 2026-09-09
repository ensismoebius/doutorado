"""Wavelet Lab dock (FIXME §10, §55).

Takes the currently selected signal, runs the repository's own
wavelet-packet transform through ``nn_microscope.wavelet``, and shows the
leaves with per-band energy. Selecting a leaf plots its coefficients — one
step on the "why did this number become this number?" path from a feature
back to the raw signal.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from experiment_microscope.core.selection import SelectionState
from experiment_microscope.views._help import HelpBox
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg
from experiment_microscope.views._timesync import TimeCursor

_WAVELETS = ["haar", "daub4", "daub6", "daub8", "daub10", "daub12", "daub20"]


_HELP = """
<b>What this shows.</b> The signal split into frequency <b>sub-bands</b> by a
<b>wavelet packet</b> decomposition (a binary tree of filters; each leaf is a
narrow frequency range). Unlike a plain spectrum this keeps <i>when</i> things
happen, not only <i>what</i> frequencies are present.
<br><br>
<b>Controls.</b> <i>wavelet</i> picks the filter shape (Haar = blocky and fast;
daub4…daub20 = progressively smoother Daubechies filters). <i>level</i> = tree
depth, so 2<sup>level</sup> leaves. <i>channel</i> chooses which EEG channel to
decompose.
<br><br>
<b>Table / plot.</b> Per leaf: <b>energy</b> (sum of squared coefficients = power
in that band) and <b>relative energy</b> (share of the total, bands sum to 1).
Click a leaf to see its raw coefficients.
"""



class WaveletLab(QWidget):
    def __init__(
        self,
        repo: DataRepository,
        selection: SelectionState | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._selection = selection
        self._cursor = None
        self.repo = repo
        self._signal: Signal1D | None = None
        self._decomp = None

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Wavelet Lab', _HELP))
        controls = QHBoxLayout()
        form = QFormLayout()
        self._wavelet = QComboBox()
        self._wavelet.addItems(_WAVELETS)
        self._wavelet.setCurrentText("haar")
        self._level = QSpinBox()
        self._level.setRange(1, 8)
        self._level.setValue(4)
        self._mode = QComboBox()
        self._mode.addItems(["packet", "regular"])
        self._channel = QComboBox()  # populated when the input is multichannel (§8)
        self._channel.currentIndexChanged.connect(self._recompute)
        for w in (self._wavelet, self._level, self._mode):
            w.currentIndexChanged.connect(self._recompute) if isinstance(w, QComboBox) else \
                w.valueChanged.connect(self._recompute)
        form.addRow("wavelet", self._wavelet)
        form.addRow("level", self._level)
        form.addRow("mode", self._mode)
        form.addRow("channel", self._channel)
        controls.addLayout(form)
        self._status = QLabel("No signal selected.")
        self._status.setWordWrap(True)
        controls.addWidget(self._status, 1)
        root.addLayout(controls)

        if not PG_OK:
            root.addWidget(missing_widget("Wavelet Lab"))
            self._tree = None
            return

        split = QSplitter(Qt.Orientation.Horizontal)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Leaf", "Energy", "Rel. energy"])
        self._tree.setColumnWidth(0, 90)
        self._tree.currentItemChanged.connect(self._on_leaf)
        split.addWidget(self._tree)
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        split.addWidget(self._plot)
        if selection is not None:
            self._cursor = TimeCursor(self._plot.getPlotItem(), selection)
        split.setSizes([200, 500])
        root.addWidget(split, 1)

    # -- publication export (§30) ------------------------------
    def can_export(self) -> bool:
        return self._decomp is not None

    def export_figure(self, path, **opts):
        from experiment_microscope.viz.mpl_export import annotate_provenance, new_figure, save_figure

        d = self._decomp
        fig = new_figure(width_in=opts.get("width_in", 6.5), height_in=opts.get("height_in", 3.0))
        ax = fig.add_subplot(111)
        e = np.asarray(d.subband_energies, dtype=float)
        ax.bar(np.arange(e.size), e, width=0.85)
        ax.set_xlabel("sub-band" if d.packet else "detail level")
        ax.set_ylabel("energy (RMS)")
        ax.set_title(f"{d.mode} {d.wavelet}, level {d.levels} — {self._signal.label or ''}")
        annotate_provenance(ax, f"origin: computed (nn_microscope.wavelet); {e.size} bands")
        return save_figure(fig, path, transparent=opts.get("transparent", False))

    # -- external API -------------------------------------------
    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        adapter = self.repo.adapter(adapter_key)
        try:
            self._signal = adapter.load_signal(node)
        except NotImplementedError:
            self._signal = None
            self._status.setText("Selected object has no 1-D signal to decompose.")
            return
        except BindingUnavailableError as exc:
            self._signal = None
            self._status.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._signal = None
            self._status.setText(f"load_signal failed: {exc}")
            return
        self._sync_channel_selector()
        self._recompute()

    def _sync_channel_selector(self) -> None:
        raw = None if self._signal is None else np.asarray(self._signal.samples)
        n = raw.shape[0] if (raw is not None and raw.ndim == 2) else 0
        self._channel.blockSignals(True)
        self._channel.clear()
        if n:
            names = self._signal.channel_names or tuple(f"ch{c}" for c in range(n))
            self._channel.addItems(list(names))
        self._channel.setEnabled(bool(n))
        self._channel.blockSignals(False)

    # -- internals ---------------------------------------------
    def _recompute(self) -> None:
        if self._tree is None or self._signal is None:
            return
        from experiment_microscope.processing import wavelet as wl

        raw = np.asarray(self._signal.samples, dtype=float)
        if raw.ndim == 2:  # multichannel (EEG): decompose the selected channel (§8)
            ch = max(0, self._channel.currentIndex())
            data = raw[min(ch, raw.shape[0] - 1)]
        else:
            data = raw.ravel()
        try:
            self._decomp = wl.decompose(
                data, self._wavelet.currentText(), self._mode.currentText(), self._level.value()
            )
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"decompose failed: {exc}")
            return

        energies = self._decomp.subband_energies
        total = float(energies.sum()) or 1.0
        self._tree.clear()
        for i, e in enumerate(energies):
            item = QTreeWidgetItem(
                [str(i), f"{e:.6g}", f"{100.0 * e / total:.2f}%"]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, i)
            self._tree.addTopLevelItem(item)
        self._status.setText(
            f"{self._signal.label or 'signal'}  [computed] — "
            f"{self._decomp.mode} {self._decomp.wavelet}, "
            f"{len(energies)} band(s), {data.size} samples in"
        )
        self._plot.clear()
        self._plot.plot(np.arange(data.size), data, pen=pg.mkPen((120, 170, 255)))
        self._plot.setTitle("input signal")
        if self._cursor is not None:
            self._cursor.reattach()

    def _on_leaf(self, current: QTreeWidgetItem | None, _prev) -> None:
        if current is None or self._decomp is None:
            return
        idx = current.data(0, Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        try:
            coeffs = self._decomp.leaf(int(idx))
        except RuntimeError:
            # regular (non-packet) transform — show the k-th detail band instead
            coeffs = self._decomp.transformed_signal
        self._plot.clear()
        self._plot.plot(np.arange(coeffs.size), coeffs, pen=pg.mkPen((255, 190, 90)))
        self._plot.setTitle(f"leaf {idx} coefficients ({coeffs.size})")
        if self._cursor is not None:
            self._cursor.reattach()
