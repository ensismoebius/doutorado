"""Feature Matrix inspector (FIXME §14, §29).

samples x features heatmap for the selected feature set, with per-column
statistics and an optional per-column z-score normalization toggle. Rows are
labelled with the sample's subject; clicking a cell reports its exact value.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.data.adapters import FeatureMatrix, TreeNode
from experiment_microscope.views._help import HelpBox
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg


_HELP = """
<b>What this shows.</b> The full table of handcrafted feature values for a thesis
run: one row per sample, one column per feature.
<br><br>
<b>Colours.</b> A heatmap — brighter / darker means larger / smaller value; the
colour bar gives the scale. The per-column statistics (mean, standard deviation,
min, max) sit beside it. Turning on the z-score toggle rescales <i>for display
only</i> so columns of different magnitude become comparable — the underlying
numbers do not change.
<br><br>
<b>Click a cell</b> to read its exact value, the sample's class, and (when the
wavelet layout allows) which frequency band it came from.
<br><br>
Feature names encode the descriptor and scale, e.g. energy / <b>ZCR</b>
(zero-crossing rate) / entropy / <b>Teager</b> (Teager–Kaiser energy) / jitter /
shimmer per wavelet band.
"""



class FeatureMatrixView(QWidget):
    def __init__(
        self,
        repo: DataRepository,
        selection=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.repo = repo
        self.selection = selection
        self._matrix: FeatureMatrix | None = None

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Feature Matrix', _HELP))
        from experiment_microscope.core.i18n import t as _t
        self._t = _t
        top = QHBoxLayout()
        self._normalize = QCheckBox(_t("per-column z-score (display only)"))
        # On by default: handcrafted columns span many orders of magnitude, so the
        # raw heatmap is dominated by one column and reads as a black rectangle.
        self._normalize.setChecked(True)
        self._normalize.stateChanged.connect(self._render)
        self._status = QLabel(_t("Select a Phase-00 handcrafted run."))
        self._status.setWordWrap(True)
        top.addWidget(self._normalize)
        top.addWidget(self._status, 1)
        root.addLayout(top)

        if not PG_OK:
            root.addWidget(missing_widget("Feature Matrix"))
            self._img = None
            return

        split = QSplitter(Qt.Orientation.Vertical)
        self._plot = pg.PlotWidget()
        self._img = pg.ImageItem()
        self._plot.addItem(self._img)
        self._plot.setLabel("bottom", _t("feature"))
        self._plot.setLabel("left", _t("sample"))
        self._plot.scene().sigMouseClicked.connect(self._on_click)
        split.addWidget(self._plot)
        self._stats = QTableWidget(0, 4)
        self._stats.setHorizontalHeaderLabels(
            [_t("feature"), _t("mean"), _t("std"), _t("min / max")])
        split.addWidget(self._stats)
        split.setSizes([400, 160])
        root.addWidget(split, 1)

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        if self._img is None:
            return
        adapter = self.repo.adapter(adapter_key)
        try:
            self._matrix = adapter.load_features(node)
        except NotImplementedError:
            self._matrix = None
            self._status.setText(self._t("No live feature matrix for this object."))
            return
        except BindingUnavailableError as exc:
            self._matrix = None
            self._status.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._matrix = None
            self._status.setText(self._t("load_features failed: {exc}", exc=exc))
            return
        self._render()

    def _render(self) -> None:
        if self._img is None or self._matrix is None:
            return
        m = self._matrix
        data = np.asarray(m.values, dtype=float)
        shown = data
        if self._normalize.isChecked():
            mu = data.mean(axis=0, keepdims=True)
            sd = data.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            shown = (data - mu) / sd
        # ImageItem is column-major over (x=feature, y=sample)
        self._img.setImage(shown.T, autoLevels=True)
        self._status.setText(
            self._t("{label}  [{origin}] — {r} samples x {c} features",
                    label=m.set_label, origin=m.origin.value,
                    r=data.shape[0], c=data.shape[1])
        )
        self._stats.setRowCount(data.shape[1])
        for j in range(data.shape[1]):
            col = data[:, j]
            for c, text in enumerate(
                (m.feature_names[j], f"{col.mean():.4g}", f"{col.std():.4g}",
                 f"{col.min():.3g} / {col.max():.3g}")
            ):
                self._stats.setItem(j, c, QTableWidgetItem(text))

    def can_export(self) -> bool:
        return self._matrix is not None

    def export_figure(self, path, **opts):
        from experiment_microscope.viz.mpl_export import annotate_provenance, new_figure, save_figure

        m = self._matrix
        data = np.asarray(m.values, dtype=float)
        if self._normalize.isChecked():
            mu, sd = data.mean(0, keepdims=True), data.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            data = (data - mu) / sd
        fig = new_figure(width_in=opts.get("width_in", 6.0), height_in=opts.get("height_in", 4.0))
        ax = fig.add_subplot(111)
        im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_xlabel("feature")
        ax.set_ylabel("sample")
        ax.set_title(m.set_label)
        fig.colorbar(im, ax=ax, shrink=0.8)
        annotate_provenance(ax, f"origin: {m.origin.value}; {data.shape[0]}x{data.shape[1]}")
        return save_figure(fig, path, transparent=opts.get("transparent", False))

    def _on_click(self, event) -> None:
        """Exact-value readout for the clicked cell (FIXME §29)."""
        if self._matrix is None or self._img is None:
            return
        pos = self._img.mapFromScene(event.scenePos())
        j, i = int(pos.x()), int(pos.y())  # ImageItem plotted as shown.T → x=feature, y=sample
        data = np.asarray(self._matrix.values, dtype=float)
        if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1]):
            return
        m = self._matrix
        self._status.setText(
            self._t("{name} @ {sample}  =  {val}  (class {cls})  [{origin}]",
                    name=m.feature_names[j], sample=m.sample_labels[i],
                    val=f"{data[i, j]:.6g}", cls=m.class_labels[i], origin=m.origin.value)
        )
        if self.selection is not None:
            self.selection.set("feature", j, cascade=False)
