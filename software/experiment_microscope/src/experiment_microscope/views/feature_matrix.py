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
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.processing._binding import BindingUnavailableError
from experiment_microscope.views._pg import PG_OK, missing_widget, pg


class FeatureMatrixView(QWidget):
    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._matrix: FeatureMatrix | None = None

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self._normalize = QCheckBox("per-column z-score (display only)")
        self._normalize.stateChanged.connect(self._render)
        self._status = QLabel("Select a Phase-00 handcrafted run.")
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
        self._plot.setLabel("bottom", "feature")
        self._plot.setLabel("left", "sample")
        split.addWidget(self._plot)
        self._stats = QTableWidget(0, 4)
        self._stats.setHorizontalHeaderLabels(["feature", "mean", "std", "min / max"])
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
            self._status.setText("No live feature matrix for this object.")
            return
        except BindingUnavailableError as exc:
            self._matrix = None
            self._status.setText(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._matrix = None
            self._status.setText(f"load_features failed: {exc}")
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
            f"{m.set_label}  [{m.origin.value}] — {data.shape[0]} samples x {data.shape[1]} features"
        )
        self._stats.setRowCount(data.shape[1])
        for j in range(data.shape[1]):
            col = data[:, j]
            for c, text in enumerate(
                (m.feature_names[j], f"{col.mean():.4g}", f"{col.std():.4g}",
                 f"{col.min():.3g} / {col.max():.3g}")
            ):
                self._stats.setItem(j, c, QTableWidgetItem(text))
