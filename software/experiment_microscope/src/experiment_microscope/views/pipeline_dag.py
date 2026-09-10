"""Pipeline DAG (FIXME §23).

A clickable diagram of the current experiment's processing graph. Clicking a
node opens the view that inspects that stage. The graph is *descriptive* — it
mirrors what the code does (see Experiments/Meeting01.md, Experiments/Thesis.md),
it does not execute anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

_BOX_W, _BOX_H, _COL, _ROW = 150.0, 44.0, 200.0, 90.0


@dataclass(frozen=True)
class DagNode:
    key: str
    label: str
    col: int
    row: int
    tab: str = ""  # central-tab this node opens, "" = none yet


# meeting01: FSDD window -> encode -> SNN-AE -> latent -> reconstruction -> metrics
_MEETING01 = (
    [
        DagNode("raw", "raw WAV", 0, 0, "Signal"),
        DagNode("window", "window +\nz-score", 1, 0, "Signal"),
        DagNode("encoding", "spike\nencoding", 2, 0, "Encoding Lab"),
        DagNode("wavelet", "wavelet", 2, 1, "Wavelet Lab"),
        DagNode("snn", "SNN-AE\n(sweep)", 3, 0, ""),
        DagNode("latent", "latent", 4, 0, ""),
        DagNode("recon", "reconstruction", 5, 0, ""),
        DagNode("metrics", "MSE / MAE / R²", 6, 0, ""),
    ],
    [("raw", "window"), ("window", "encoding"), ("window", "wavelet"),
     ("encoding", "snn"), ("snn", "latent"), ("latent", "recon"), ("recon", "metrics")],
)

# thesis: raw signal -> wavelet packet -> handcrafted features -> paraconsistent -> ranking -> [DSNN -> EER]
_THESIS = (
    [
        DagNode("raw", "raw EEG /\naudio", 0, 0, "Signal"),
        DagNode("wavelet", "wavelet\npacket", 1, 0, "Wavelet Lab"),
        DagNode("features", "handcrafted\nfeatures", 2, 0, "Feature Matrix"),
        DagNode("para", "paraconsistent\nα β G1 G2", 3, 0, "Paraconsistent plane"),
        DagNode("rank", "feature\nranking", 4, 0, "Paraconsistent plane"),
        DagNode("dsnn", "DSNN\nauth (phase01)", 5, 0, ""),
        DagNode("eer", "EER / AUC", 6, 0, ""),
    ],
    [("raw", "wavelet"), ("wavelet", "features"), ("features", "para"),
     ("para", "rank"), ("rank", "dsnn"), ("dsnn", "eer")],
)

_GRAPHS = {"meeting01": _MEETING01, "thesis": _THESIS}


class _NodeItem(QGraphicsRectItem):
    def __init__(self, node: DagNode, on_click) -> None:
        super().__init__(QRectF(0, 0, _BOX_W, _BOX_H))
        self.node = node
        self._on_click = on_click
        self.setPos(node.col * _COL, node.row * _ROW)
        active = bool(node.tab)
        self.setBrush(QBrush(QColor(60, 78, 96) if active else QColor(45, 45, 45)))
        self.setPen(QPen(QColor(120, 190, 140) if active else QColor(90, 90, 90),
                         1.5, Qt.PenStyle.SolidLine if active else Qt.PenStyle.DashLine))
        self.setCursor(Qt.CursorShape.PointingHandCursor if active else Qt.CursorShape.ArrowCursor)
        text = QGraphicsSimpleTextItem(node.label, self)
        text.setBrush(QBrush(QColor(230, 230, 230)))
        br = text.boundingRect()
        text.setPos((_BOX_W - br.width()) / 2, (_BOX_H - br.height()) / 2)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.node.tab:
            self._on_click(self.node.tab)
        super().mousePressEvent(event)


class PipelineDag(QWidget):
    node_activated = Signal(str)  # central-tab name

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self._view)
        self.show_experiment(None)

    def show_experiment(self, experiment: str | None) -> None:
        self._scene.clear()
        from experiment_microscope.core.i18n import t as _t
        graph = _GRAPHS.get(experiment or "")
        if graph is None:
            t = self._scene.addText(_t("Select a meeting01 or thesis object."))
            t.setDefaultTextColor(QColor(180, 180, 180))
            self._scene.setSceneRect(self._scene.itemsBoundingRect())
            return
        nodes, edges = graph
        items: dict[str, _NodeItem] = {}
        for n in nodes:
            item = _NodeItem(n, self.node_activated.emit)
            self._scene.addItem(item)
            items[n.key] = item
        pen = QPen(QColor(140, 140, 140), 1.4)
        for a, b in edges:
            ia, ib = items[a], items[b]
            p1 = ia.scenePos() + ia.boundingRect().center()
            p2 = ib.scenePos() + ib.boundingRect().center()
            path = QPainterPath(p1)
            path.lineTo(p2)
            edge = QGraphicsPathItem(path)
            edge.setPen(pen)
            edge.setZValue(-1)
            self._scene.addItem(edge)
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20))
        self._fit()

    def _fit(self) -> None:
        rect = self._scene.sceneRect()
        if rect.isValid() and not rect.isEmpty():
            self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit()  # keep the whole DAG in view when the panel is resized
