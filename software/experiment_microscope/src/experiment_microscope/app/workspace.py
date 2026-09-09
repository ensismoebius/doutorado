"""The dockable scientific workstation (FIXME §6).

A ``QMainWindow`` whose panels are all ``QDockWidget``s so the researcher can
rearrange / float / tab them and the layout is saved between sessions. The
central widget is a tab stack of the main visualizations; the explorer,
provenance inspector and (meeting01) session dashboard are docks.

All panels talk only through ``SelectionState`` (FIXME §42).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from experiment_microscope.core.animation import TimelinePlayer
from experiment_microscope.core.selection import SelectionState
from experiment_microscope.core.state import AppState
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.explorer import ExplorerTree
from experiment_microscope.views.paraconsistent_plane import ParaconsistentPlane
from experiment_microscope.views.provenance_inspector import ProvenanceInspector
from experiment_microscope.views.signal_view import SignalView
from experiment_microscope.views.wavelet_lab import WaveletLab

_ORG = "doutorado"
_APP = "experiment_microscope"


class Workspace(QMainWindow):
    def __init__(self, initial_experiment: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Experiment Microscope")
        self.resize(1400, 900)

        self.repo = DataRepository()
        self.selection = SelectionState(self)
        self.app_state = AppState()
        self.timeline = TimelinePlayer(self)

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_statusbar()
        self._wire()

        self._restore_layout()
        if initial_experiment:
            self.explorer.select_experiment(initial_experiment)

    # -- construction ------------------------------------------------
    def _build_central(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.signal_view = SignalView(self.repo)
        self.wavelet_lab = WaveletLab(self.repo)
        self.para_plane = ParaconsistentPlane(self.repo)
        self.tabs.addTab(self.signal_view, "Signal")
        self.tabs.addTab(self.wavelet_lab, "Wavelet Lab")
        self.tabs.addTab(self.para_plane, "Paraconsistent plane")
        self.setCentralWidget(self.tabs)

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock::{title}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        self.explorer = ExplorerTree(self.repo)
        self._dock("Data Explorer", self.explorer, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.provenance = ProvenanceInspector(self.repo)
        self._dock("Inspector", self.provenance, Qt.DockWidgetArea.RightDockWidgetArea)

        self.session_log = QPlainTextEdit()
        self.session_log.setReadOnly(True)
        self.session_log.setPlainText("No meeting01 run detected under results/meeting01/.")
        self._dock("Meeting01 session", self.session_log, Qt.DockWidgetArea.BottomDockWidgetArea)

    def _build_menus(self) -> None:
        view_menu = self.menuBar().addMenu("&View")
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        refresh = QAction("Refresh paraconsistent plane", self)
        refresh.triggered.connect(self.para_plane.refresh)
        view_menu.addAction(refresh)

        exp_menu = self.menuBar().addMenu("&Experiment")
        for key, label in (("meeting01", "Meeting01"), ("thesis", "Thesis"),
                           ("paraconsistent_ga", "Paraconsistent GA")):
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, k=key: self.explorer.select_experiment(k))
            exp_menu.addAction(act)

    def _build_statusbar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self._status_selection = QLabel("—")
        self._status_res = QLabel("FULL RESOLUTION")
        bar.addWidget(self._status_selection, 1)
        bar.addPermanentWidget(self._status_res)

    # -- wiring ----------------------------------------------------
    def _wire(self) -> None:
        self.explorer.node_selected.connect(self._on_node_selected)
        self.para_plane.point_clicked.connect(self._on_para_point)
        self.app_state.display_downsampled_changed.connect(
            lambda on: self._status_res.setText("DISPLAY-DOWNSAMPLED" if on else "FULL RESOLUTION")
        )
        self.repo.cache.failed.connect(self._on_cache_failed)

    def _on_node_selected(self, node: TreeNode, adapter_key: str) -> None:
        self.selection.set("experiment", adapter_key)
        h = getattr(node, "handle", {}) or {}
        self.selection.update(
            dataset=h.get("dataset"),
            model=h.get("architecture") and f"snn-ae/{h.get('architecture')}",
            encoding=h.get("encoding"),
        )
        self._status_selection.setText(f"{adapter_key} › {node.kind} › {node.label}")
        self.provenance.show_node(node, adapter_key)
        self.signal_view.show_node(node, adapter_key)
        self.wavelet_lab.show_node(node, adapter_key)
        if adapter_key == "meeting01":
            self._refresh_session_log()

    def _on_para_point(self, point) -> None:
        facet = getattr(point, "facet", {}) or {}
        self._status_selection.setText(f"paraconsistent › {point.label}")
        if facet.get("run_tag"):
            self.selection.update(experiment="thesis")

    def _on_cache_failed(self, _key, exc, _tb) -> None:
        self.statusBar().showMessage(f"compute failed: {exc}", 8000)

    def _refresh_session_log(self) -> None:
        adapter = self.repo.adapter("meeting01")
        text = None
        try:
            text = adapter.dashboard_text()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            text = f"session dashboard unavailable: {exc}"
        self.session_log.setPlainText(text or "No meeting01 run detected under results/meeting01/.")

    # -- layout persistence -------------------------------------
    def _restore_layout(self) -> None:
        settings = QSettings(_ORG, _APP)
        geo = settings.value("geometry")
        state = settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)

    def closeEvent(self, event) -> None:  # noqa: N802
        settings = QSettings(_ORG, _APP)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        super().closeEvent(event)
