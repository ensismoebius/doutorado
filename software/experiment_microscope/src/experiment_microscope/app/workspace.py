"""The dockable scientific workstation (FIXME §6).

A ``QMainWindow`` whose panels are all ``QDockWidget``s so the researcher can
rearrange / float / tab them and the layout is saved between sessions. The
central widget is a tab stack of the main visualizations; the explorer,
provenance inspector and (meeting01) session dashboard are docks.

All panels talk only through ``SelectionState`` (FIXME §42).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDockWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.animation import TimelinePlayer
from experiment_microscope.core.bookmarks import BookmarkStore, make_bookmark
from experiment_microscope.core.selection import SelectionState
from experiment_microscope.core.state import AppState
from experiment_microscope.core.theme import apply_theme
from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
import numpy as np

from experiment_microscope.views.artifact_inspector import ArtifactInspector
from experiment_microscope.views.bookmarks_dock import BookmarksDock
from experiment_microscope.views.comparison_view import ComparisonView
from experiment_microscope.views.developer_panel import DeveloperPanel
from experiment_microscope.views.explorer import ExplorerTree
from experiment_microscope.views.paraconsistent_landscape import ParaconsistentLandscape
from experiment_microscope.views.paraconsistent_plane import ParaconsistentPlane
from experiment_microscope.views.nsga_view import NsgaView
from experiment_microscope.views.pipeline_dag import PipelineDag
from experiment_microscope.views.feature_matrix import FeatureMatrixView
from experiment_microscope.views.encoding_lab import EncodingLab
from experiment_microscope.views.experiment_timeline import ExperimentTimeline
from experiment_microscope.views.follow_data import FollowDataBar
from experiment_microscope.views.latent_explorer import LatentExplorer
from experiment_microscope.views.provenance_inspector import ProvenanceInspector
from experiment_microscope.views.reconstruction_view import ReconstructionView
from experiment_microscope.views.reproduce_panel import ReproducePanel
from experiment_microscope.views.search_bar import SearchBar
from experiment_microscope.views.snn_lab import SnnLab
from experiment_microscope.views.snn_3d import Snn3D
from experiment_microscope.views.ranking_view import RankingView
from experiment_microscope.views.signal_view import SignalView
from experiment_microscope.views.transport_bar import TransportBar
from experiment_microscope.views.triangle_view import TriangleView
from experiment_microscope.views.wavelet_3d import Wavelet3D
from experiment_microscope.views.wavelet_lab import WaveletLab

_ORG = "doutorado"
_APP = "experiment_microscope"


class Workspace(QMainWindow):
    def __init__(
        self,
        initial_experiment: str | None = None,
        initial_tour: str | None = None,
        initial_tab: str | None = None,
    ) -> None:
        super().__init__()
        self.resize(1400, 900)
        self._clamp_to_screen()

        from experiment_microscope.core.i18n import language as _lang, set_language, t

        set_language(str(QSettings(_ORG, _APP).value("language", _lang())))
        self.setWindowTitle(t("Experiment Microscope"))

        self.repo = DataRepository()
        self.selection = SelectionState(self)
        self.app_state = AppState()
        self.timeline = TimelinePlayer(self)

        self._build_central()
        self._build_docks()
        self._build_menus()
        self._build_statusbar()
        self._build_shortcuts()
        self._wire()

        had_layout = self._restore_layout()
        self._clamp_to_screen()
        self._apply_dock_sizes(force=not had_layout)
        saved_theme = QSettings(_ORG, _APP).value("theme", "system")
        self._apply_theme(str(saved_theme), persist=False)
        if initial_experiment:
            self.explorer.select_experiment(initial_experiment)
        if initial_tour:
            self.explorer.select_experiment(initial_tour)
            self._start_tour()
        if initial_tab:
            self._open_tab(initial_tab)

    # -- construction ------------------------------------------------
    def _build_central(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.signal_view = SignalView(self.repo, self.selection, self.app_state)
        self.wavelet_lab = WaveletLab(self.repo, self.selection)
        self.wavelet_3d = Wavelet3D(self.repo, self.app_state)
        self.feature_matrix = FeatureMatrixView(self.repo, self.selection)
        self.encoding_lab = EncodingLab(self.repo, self.selection)
        self.snn_lab = SnnLab(self.repo, self.selection)
        self.snn_3d = Snn3D(self.repo, self.app_state)
        self.snn_3d.set_timeline(self.timeline)
        self.latent_explorer = LatentExplorer(self.repo, self.app_state)
        self.latent_explorer.sample_activated.connect(self._on_latent_sample)
        self.latent_explorer.working.connect(
            lambda on: self.busy.begin("Projecting the latent space…") if on else self.busy.end())
        self.para_plane = ParaconsistentPlane(self.repo)
        self.para_landscape = ParaconsistentLandscape(self.repo)
        self.pipeline_dag = PipelineDag()
        self.triangle = TriangleView(self.repo)
        self.triangle.set_timeline(self.timeline)
        self.comparison = ComparisonView(self.repo)
        self.reconstruction = ReconstructionView(self.repo)
        self.nsga = NsgaView(self.repo)
        self.ranking = RankingView(self.repo)
        self.ranking.run_activated.connect(self._on_ranking_run)
        self.timeline_view = ExperimentTimeline(self.repo)
        self.timeline_view.config_activated.connect(self._on_timeline_config)
        self.pipeline_dag.node_activated.connect(self._open_tab)
        for widget, key in (
            (self.signal_view, "Signal"),
            (self.wavelet_lab, "Wavelet Lab"),
            (self.wavelet_3d, "Wavelet 3D"),
            (self.feature_matrix, "Feature Matrix"),
            (self.encoding_lab, "Encoding Lab"),
            (self.snn_lab, "SNN Lab"),
            (self.snn_3d, "SNN 3D"),
            (self.latent_explorer, "Latent Space"),
            (self.reconstruction, "Reconstruction"),
            (self.para_plane, "Paraconsistent plane"),
            (self.para_landscape, "Paraconsistent landscape"),
            (self.pipeline_dag, "Pipeline"),
            (self.triangle, "Triangle"),
            (self.comparison, "Comparison"),
            (self.nsga, "NSGA-II"),
            (self.ranking, "Ranking"),
            (self.timeline_view, "Timeline"),
        ):
            i = self.tabs.addTab(widget, key)
            self.tabs.tabBar().setTabData(i, key)
        self.para_landscape.point_clicked.connect(self._on_para_point)
        # don't let a single wide view (e.g. Comparison's table) push the window
        # past the screen — let it clip/scroll within the tab instead.
        self.tabs.setMinimumWidth(360)
        self._retranslate_tabs()

        self.follow_bar = FollowDataBar(self.repo)
        self.follow_bar.stage_activated.connect(self._open_tab)
        self.transport = TransportBar(self.timeline)

        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        # The pipeline strip and colour key are wider than a small screen; scroll
        # them sideways rather than forcing the whole window past the monitor edge.
        self._follow_scroll = self._hscroll(self.follow_bar)
        col.addWidget(self._follow_scroll)
        self._legend_host = QWidget()
        _lh = QVBoxLayout(self._legend_host)
        _lh.setContentsMargins(0, 0, 0, 0)
        _lh.setSpacing(0)
        self._legend_scroll = self._hscroll(self._legend_host)
        col.addWidget(self._legend_scroll)
        col.addWidget(self.tabs, 1)
        col.addWidget(self.transport)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._refresh_legend)
        self._refresh_legend(self.tabs.currentIndex())

    def _hscroll(self, inner: QWidget) -> QScrollArea:
        """Wrap a too-wide horizontal strip so it scrolls instead of stretching
        the window past the screen. Height tracks the inner widget."""
        sa = QScrollArea()
        sa.setWidget(inner)
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setSizeAdjustPolicy(QScrollArea.SizeAdjustPolicy.AdjustToContents)
        hint = inner.sizeHint().height()
        sa.setMinimumHeight(hint)
        sa.setMaximumHeight(hint + 16)  # room for a scrollbar when it appears
        sa.setMinimumWidth(0)
        return sa

    def _tab_key(self, i: int) -> str:
        """Stable English key for a tab (its visible text is translated)."""
        data = self.tabs.tabBar().tabData(i)
        return str(data) if data is not None else self.tabs.tabText(i)

    def _retranslate_tabs(self) -> None:
        from experiment_microscope.core.i18n import t

        for i in range(self.tabs.count()):
            self.tabs.setTabText(i, t(self._tab_key(i)))

    def _refresh_legend(self, idx: int) -> None:
        """Swap the colour-key strip under the tab bar to match the active tab."""
        from experiment_microscope.core import palette

        host = self._legend_host.layout()
        while host.count():
            w = host.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        strip = palette.legend_strip(self._tab_key(idx) if idx >= 0 else "")
        host.addWidget(strip)

    def _open_tab(self, name: str) -> None:
        for i in range(self.tabs.count()):
            if self._tab_key(i) == name:
                self.tabs.setTabVisible(i, True)
                self.tabs.setCurrentIndex(i)
                return

    #: which tabs make sense for a given selection. value(handle, adapter) -> bool.
    #: tabs not listed here are always shown (Comparison, Ranking, Pipeline).
    _TAB_RULES = {
        "Signal": lambda h, a: (a == "meeting01" and h.get("level") == "window")
        or (a == "thesis" and h.get("level") == "sample"),
        "Wavelet Lab": lambda h, a: (a == "meeting01" and h.get("level") == "window")
        or (a == "thesis" and h.get("level") == "sample"),
        "Wavelet 3D": lambda h, a: (a == "meeting01" and h.get("level") == "window")
        or (a == "thesis" and h.get("level") == "sample"),
        "Feature Matrix": lambda h, a: a == "thesis" and h.get("level") == "run",
        "Triangle": lambda h, a: a == "thesis" and h.get("level") == "run",
        "Encoding Lab": lambda h, a: a == "meeting01" and h.get("level") == "window",
        "SNN Lab": lambda h, a: a == "meeting01" and h.get("level") == "window",
        "SNN 3D": lambda h, a: a == "meeting01" and h.get("level") == "window",
        "Reconstruction": lambda h, a: a == "meeting01" and h.get("level") == "window",
        "Latent Space": lambda h, a: a == "meeting01",
        "Timeline": lambda h, a: a == "meeting01",
        "Paraconsistent plane": lambda h, a: a in ("thesis", "paraconsistent_ga"),
        "Paraconsistent landscape": lambda h, a: a in ("thesis", "paraconsistent_ga"),
        "NSGA-II": lambda h, a: a == "paraconsistent_ga",
    }

    def _update_tab_visibility(self, handle: dict, adapter_key: str) -> None:
        """Show only the tabs that apply to the current selection; keep the
        always-on ones. If the active tab is hidden, fall back to the first
        visible one."""
        cur = self.tabs.currentIndex()
        for i in range(self.tabs.count()):
            rule = self._TAB_RULES.get(self._tab_key(i))
            self.tabs.setTabVisible(i, True if rule is None else bool(rule(handle, adapter_key)))
        if cur >= 0 and not self.tabs.isTabVisible(cur):
            for i in range(self.tabs.count()):
                if self.tabs.isTabVisible(i):
                    self.tabs.setCurrentIndex(i)
                    break

    def _dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        from experiment_microscope.core.i18n import t

        dock = QDockWidget(t(title), self)
        dock.setObjectName(f"dock::{title}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self) -> None:
        self.explorer = ExplorerTree(self.repo)
        self.search_bar = SearchBar(self.explorer)
        self.search_bar.path_activated.connect(self.explorer.select_path)
        explorer_panel = QWidget()
        _pl = QVBoxLayout(explorer_panel)
        _pl.setContentsMargins(0, 0, 0, 0)
        _pl.setSpacing(2)
        _pl.addWidget(self.search_bar)
        _pl.addWidget(self.explorer, 1)
        explorer_panel.setMinimumWidth(150)
        self._left_dock = self._dock(
            "Data Explorer", explorer_panel, Qt.DockWidgetArea.LeftDockWidgetArea)

        self.provenance = ProvenanceInspector(self.repo)
        prov_dock = self._dock("Inspector", self.provenance, Qt.DockWidgetArea.RightDockWidgetArea)

        self.artifact_inspector = ArtifactInspector(self.repo)
        art_dock = self._dock("Raw artifact", self.artifact_inspector, Qt.DockWidgetArea.RightDockWidgetArea)
        self.reproduce_panel = ReproducePanel(self.repo)
        repro_dock = self._dock("Reproduce", self.reproduce_panel, Qt.DockWidgetArea.RightDockWidgetArea)

        self.session_log = QPlainTextEdit()
        self.session_log.setReadOnly(True)
        self.session_log.setMaximumHeight(140)
        from experiment_microscope.core.i18n import t as _t
        self.session_log.setPlainText(_t("No meeting01 run detected under results/meeting01/."))
        self._dock("Meeting01 session", self.session_log, Qt.DockWidgetArea.BottomDockWidgetArea)

        self.bookmarks = BookmarksDock()
        self.bookmarks.save_requested.connect(self._save_bookmark)
        self.bookmarks.restore_requested.connect(self._restore_bookmark)
        bm_dock = self._dock("Bookmarks", self.bookmarks, Qt.DockWidgetArea.RightDockWidgetArea)

        self.developer = DeveloperPanel(self.repo.cache, self._probe_shapes)
        dev_dock = self._dock("Developer", self.developer, Qt.DockWidgetArea.RightDockWidgetArea)
        dev_dock.setVisible(False)  # opt-in (FIXME §37)

        # One tab group on the right so the side panels never fight for width
        # on a small screen (FIXME §6 — the docks must stay usable).
        self._right_docks = [prov_dock, art_dock, repro_dock, bm_dock, dev_dock]
        for d in self._right_docks:
            d.widget().setMinimumWidth(180)
        for prev, nxt in zip(self._right_docks, self._right_docks[1:]):
            self.tabifyDockWidget(prev, nxt)
        prov_dock.raise_()

    def _build_menus(self) -> None:
        from experiment_microscope.core.i18n import LANGUAGES, language, set_language, t

        view_menu = self.menuBar().addMenu(t("&View"))
        for dock in self.findChildren(QDockWidget):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addSeparator()
        refresh = QAction(t("Refresh paraconsistent views"), self)
        refresh.triggered.connect(self.para_plane.refresh)
        refresh.triggered.connect(self.para_landscape.refresh)
        view_menu.addAction(refresh)

        reset_layout = QAction(t("Reset panel layout"), self)
        reset_layout.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout)

        theme_menu = view_menu.addMenu(t("Theme"))
        self._theme_group = QActionGroup(self)
        for key, label in (("system", t("System")), ("light", t("Light")), ("dark", t("Dark"))):
            act = QAction(label, self, checkable=True)
            act.setChecked(self.app_state.theme == key)
            act.triggered.connect(lambda _=False, k=key: self._apply_theme(k))
            self._theme_group.addAction(act)
            theme_menu.addAction(act)

        lang_menu = view_menu.addMenu(t("Language"))
        self._lang_group = QActionGroup(self)
        for code, label in LANGUAGES.items():
            act = QAction(label, self, checkable=True)
            act.setChecked(language() == code)
            act.triggered.connect(lambda _=False, c=code: self._set_language(c))
            self._lang_group.addAction(act)
            lang_menu.addAction(act)
        view_menu.addSeparator()

        self._low_perf_action = QAction(t("Low-performance mode"), self)
        self._low_perf_action.setCheckable(True)
        self._low_perf_action.setChecked(self.app_state.low_performance_mode)
        self._low_perf_action.toggled.connect(self._set_low_performance)
        view_menu.addAction(self._low_perf_action)

        self._presentation_action = QAction(t("Presentation mode"), self)
        self._presentation_action.setCheckable(True)
        self._presentation_action.setChecked(self.app_state.presentation_mode)
        self._presentation_action.setShortcut(QKeySequence(Qt.Key.Key_F5))
        self._presentation_action.toggled.connect(self._set_presentation_mode)
        view_menu.addAction(self._presentation_action)

        export_menu = self.menuBar().addMenu(t("E&xport"))
        act = QAction(t("Export current view…"), self)
        act.triggered.connect(self._export_current_view)
        export_menu.addAction(act)

        exp_menu = self.menuBar().addMenu(t("&Experiment"))
        for key, label in (("meeting01", "Meeting01"), ("thesis", "Thesis"),
                           ("paraconsistent_ga", "Paraconsistent GA")):
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, k=key: self.explorer.select_experiment(k))
            exp_menu.addAction(act)

        help_menu = self.menuBar().addMenu(t("&Help"))
        tour = QAction(t("▶  Start guided tour"), self)
        tour.setShortcut(QKeySequence("Ctrl+G"))
        tour.triggered.connect(self._start_tour)
        help_menu.addAction(tour)
        help_menu.addSeparator()
        explain = QAction(t("Explain the current view"), self)
        explain.setShortcut(QKeySequence("F1"))
        explain.triggered.connect(self._explain_current_view)
        help_menu.addAction(explain)
        gloss = QAction(t("Glossary (all terms)…"), self)
        gloss.triggered.connect(self._show_glossary)
        help_menu.addAction(gloss)

    def _retranslate_menus(self) -> None:
        self.menuBar().clear()
        self._build_menus()

    def _set_language(self, code: str) -> None:
        from experiment_microscope.core.i18n import set_language, t

        set_language(code)
        QSettings(_ORG, _APP).setValue("language", code)
        self._retranslate_tabs()
        self._retranslate_menus()
        self._refresh_legend(self.tabs.currentIndex())
        node = getattr(self, "_current_node", None)
        if node is not None:
            self._on_node_selected(node, self._current_adapter)
        self.statusBar().showMessage(
            t("Language changed — some fixed labels update after a restart."), 6000)

    # -- presenter ergonomics (inspired by efficient_nn_lab) --------
    def _build_shortcuts(self) -> None:
        """Keyboard-remote friendly navigation for lecturing (FIXME §51).

        A presenter clicker sends PageUp / PageDown; Space / arrows drive the
        guided tour when it is open, otherwise the animation transport.
        """
        specs = (
            (Qt.Key.Key_Space, self._presenter_advance),
            (Qt.Key.Key_Right, self._presenter_advance),
            (Qt.Key.Key_PageDown, self._presenter_advance),
            (Qt.Key.Key_Left, self._presenter_back),
            (Qt.Key.Key_PageUp, self._presenter_back),
            (Qt.Key.Key_Escape, self._presenter_escape),
        )
        self._shortcuts = []
        for key, slot in specs:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)
            self._shortcuts.append(sc)

    def _typing_focus(self) -> bool:
        w = QApplication.focusWidget()
        return isinstance(w, (QLineEdit, QAbstractSpinBox))

    def _tour_open(self) -> bool:
        return bool(getattr(self, "_tour_active", False))

    def _presenter_advance(self) -> None:
        if self._typing_focus():
            return
        if self._tour_open():
            self._story_panel._advance()
        elif self.timeline.total_frames > 1:
            self.timeline.step_forward()

    def _presenter_back(self) -> None:
        if self._typing_focus():
            return
        if self._tour_open():
            self._story_panel._prev()
        elif self.timeline.total_frames > 1:
            self.timeline.step_backward()

    def _presenter_escape(self) -> None:
        if self._tour_open():
            self._end_tour()
        elif self.app_state.presentation_mode:
            self._set_presentation_mode(False)

    def _set_presentation_mode(self, on: bool) -> None:
        """F5 — full-screen lecture surface (inspired by efficient_nn_lab's
        lecture mode): hide the chrome, open the current view's explanation,
        enlarge the type. F5 / Esc restores everything."""
        self.app_state.presentation_mode = on
        act = getattr(self, "_presentation_action", None)
        if act is not None and act.isChecked() != on:
            act.setChecked(on)

        self.menuBar().setVisible(not on)
        self.statusBar().setVisible(not on)
        self._follow_scroll.setVisible(not on)
        self._legend_scroll.setVisible(not on)
        self.transport.setVisible(
            not on and not self.app_state.low_performance_mode
            and self.timeline.total_frames > 1
        )
        if not hasattr(self, "_hidden_docks"):
            self._hidden_docks = []
        if on:
            self._hidden_docks = [
                d for d in self.findChildren(QDockWidget) if d.isVisible()
            ]
            for d in self._hidden_docks:
                d.setVisible(False)
        else:
            for d in self._hidden_docks:
                d.setVisible(True)
            self._hidden_docks = []

        app = QApplication.instance()
        if app is not None:
            f = app.font()
            base = getattr(self, "_base_point_size", None)
            if base is None:
                self._base_point_size = base = max(f.pointSize(), 9)
            f.setPointSize(round(base * 1.2) if on else base)
            app.setFont(f)

        if on:
            self.showFullScreen()
            self._explain_current_view()
        else:
            self.showNormal()
            self._clamp_to_screen()

    def _start_tour(self) -> None:
        """Ctrl+G — open the Guided Tour dock for the current experiment."""
        from experiment_microscope.views.story_panel import StoryPanel

        if getattr(self, "_story_dock", None) is None:
            self._story_panel = StoryPanel(self)
            self._story_panel.finished.connect(self._end_tour)
            self._story_dock = self._dock(
                "Guided tour", self._story_panel, Qt.DockWidgetArea.LeftDockWidgetArea)
            self._story_dock.setMinimumWidth(340)
            # full-height left panel, tabbed with the Data Explorer so it is big
            # enough to read and one click gets back to the tree.
            explorer_dock = next((d for d in self.findChildren(QDockWidget)
                                  if d.objectName() == "dock::Data Explorer"), None)
            if explorer_dock is not None:
                self.tabifyDockWidget(explorer_dock, self._story_dock)
        key = getattr(self, "_current_adapter", None)
        key = key if key in ("meeting01", "thesis") else "meeting01"
        self._story_dock.setVisible(True)
        self._story_dock.raise_()
        self._story_panel.start(key)
        self._tour_active = True

    def _end_tour(self) -> None:
        self._tour_active = False
        if getattr(self, "_story_dock", None) is not None:
            self._story_dock.setVisible(False)
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, True)
        from experiment_microscope.core.i18n import t
        self.statusBar().showMessage(
            t("Tour ended — every tab is back. Explore freely."), 4000)

    def _explain_current_view(self) -> None:
        """F1 — open the current tab's 'How to read this' box, if it has one."""
        from experiment_microscope.core.i18n import t

        view = self.tabs.currentWidget()
        for child in view.findChildren(QWidget):
            if child.__class__.__name__ == "HelpBox":
                child.open()
                self.statusBar().showMessage(t("opened the explanation for this view"), 3000)
                return
        self.statusBar().showMessage(
            t("this view has no dedicated explanation yet — see Help → Glossary"), 4000)

    def _show_glossary(self) -> None:
        from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout as _VB

        from experiment_microscope.core.glossary import glossary_html
        from experiment_microscope.core.i18n import t

        dlg = QDialog(self)
        dlg.setWindowTitle(t("Glossary — every abbreviation and metric"))
        dlg.resize(560, 640)
        lay = _VB(dlg)
        browser = QTextBrowser()
        browser.setHtml(glossary_html())
        browser.setOpenExternalLinks(False)
        lay.addWidget(browser)
        dlg.show()

    def _build_statusbar(self) -> None:
        from experiment_microscope.views._busy import BusyIndicator
        from experiment_microscope.core.i18n import t

        bar = QStatusBar()
        self.setStatusBar(bar)
        self._status_selection = QLabel("—")
        self._status_res = QLabel(t("FULL RESOLUTION"))
        self.busy = BusyIndicator()
        bar.addWidget(self._status_selection, 1)
        bar.addPermanentWidget(self.busy)
        bar.addPermanentWidget(self._status_res)
        self.repo.cache.busy_changed.connect(self.busy.on_count)

    # -- wiring ----------------------------------------------------
    def _wire(self) -> None:
        from experiment_microscope.core.i18n import t

        self.explorer.node_selected.connect(self._on_node_selected)
        self.para_plane.point_clicked.connect(self._on_para_point)
        self.app_state.display_downsampled_changed.connect(
            lambda on: self._status_res.setText(
                t("DISPLAY-DOWNSAMPLED") if on else t("FULL RESOLUTION"))
        )
        self.repo.cache.failed.connect(self._on_cache_failed)
        self.selection.changed.connect(self._on_selection_changed)
        # Populate the two persisted-score scatters up front so their tabs are
        # never blank; "Refresh paraconsistent views" re-scans on demand.
        self.para_plane.refresh()
        self.para_landscape.refresh()

    def _on_selection_changed(self, field: str) -> None:
        if field == "timestep":
            from experiment_microscope.core.i18n import t
            ts = self.selection.get("timestep")
            self._status_res.setText(
                t("FULL RESOLUTION") if ts is None else t("cursor @ sample {ts}", ts=ts)
            )
        elif field == "time_range":
            tr = self.selection.get("time_range")
            if tr:
                self._status_selection.setText(
                    f"{self._status_selection.text().split('  |  ')[0]}  |  range {tr[0]}–{tr[1]}"
                )

    def _on_node_selected(self, node: TreeNode, adapter_key: str) -> None:
        self._current_node = node
        self._current_adapter = adapter_key
        self.selection.set("experiment", adapter_key)
        h = getattr(node, "handle", {}) or {}
        self.selection.update(
            dataset=h.get("dataset"),
            model=h.get("architecture") and f"snn-ae/{h.get('architecture')}",
            encoding=h.get("encoding"),
        )
        self._status_selection.setText(f"{adapter_key} › {node.kind} › {node.label}")
        self.follow_bar.update_for(node, adapter_key)
        self.pipeline_dag.show_experiment(adapter_key)
        from experiment_microscope.core.i18n import t
        with self.busy.working(t("Loading data…")):
            self.provenance.show_node(node, adapter_key)
            self.artifact_inspector.show_node(node, adapter_key)
            self.reproduce_panel.show_node(node, adapter_key)
            self.signal_view.show_node(node, adapter_key)
            self.wavelet_lab.show_node(node, adapter_key)
            self.wavelet_3d.show_node(node, adapter_key)
            self.feature_matrix.show_node(node, adapter_key)
            self.encoding_lab.show_node(node, adapter_key)
            self.snn_lab.show_node(node, adapter_key)
            self.snn_3d.show_node(node, adapter_key)
            self.latent_explorer.show_node(node, adapter_key)
            self.reconstruction.show_node(node, adapter_key)
            self.triangle.show_node(node, adapter_key)
            self.nsga.show_node(node, adapter_key)
        self._update_tab_visibility(h, adapter_key)
        if adapter_key == "meeting01":
            self._refresh_session_log()
            self.timeline_view.refresh()

    def _apply_theme(self, name: str, *, persist: bool = True) -> None:
        """FIXME §34 — dark / light / system."""
        app = QApplication.instance()
        if app is None:
            return
        eff = apply_theme(app, name)
        self.app_state.theme = name
        if persist:
            QSettings(_ORG, _APP).setValue("theme", name)
        for act in getattr(self, "_theme_group", QActionGroup(self)).actions():
            act.setChecked(act.text().lower() == name)
        self._status_selection.setText(f"theme: {name} ({eff})")
        node = getattr(self, "_current_node", None)
        if node is not None:  # restyle pyqtgraph plots on the next render
            self._on_node_selected(node, self._current_adapter)

    def _set_low_performance(self, on: bool) -> None:
        """FIXME §38 — disable 3D / animation / live updates on weak hardware."""
        self.app_state.low_performance_mode = on
        self.transport.setEnabled(not on and self.timeline.total_frames > 1)
        from experiment_microscope.core.i18n import t
        self._status_res.setText(t("LOW-PERFORMANCE MODE") if on else t("FULL RESOLUTION"))
        node = getattr(self, "_current_node", None)
        if node is not None:  # re-render so the 3D panel picks up the flag
            self.wavelet_3d.show_node(node, self._current_adapter)
            self.snn_3d.show_node(node, self._current_adapter)

    def _on_timeline_config(self, dataset: str, fold: int, config_id: str) -> None:
        self.selection.update(experiment="meeting01", dataset=dataset)
        self._status_selection.setText(
            f"timeline → meeting01 › {dataset} › fold {fold} › {config_id}"
        )

    def _on_latent_sample(self, dataset: str, fold: int, split: str, row: int) -> None:
        """A latent-explorer point → select that window everywhere (FIXME §19)."""
        adapter = self.repo.adapter("meeting01")
        try:
            meta = adapter._split(dataset, fold)[f"{split}_meta"][row]
        except Exception:  # noqa: BLE001
            meta = {}
        node = TreeNode(
            kind="sample",
            label=f"win {meta.get('window_id', row)}  spk {meta.get('speaker', '?')}  "
            f"digit {meta.get('digit', '?')}",
            handle={"level": "window", "dataset": dataset, "cv_fold": fold,
                    "split": split, "row": row},
            metadata=dict(meta),
        )
        self._on_node_selected(node, "meeting01")
        self.tabs.setCurrentWidget(self.latent_explorer)

    def _on_ranking_run(self, experiment: str, run_tag: str) -> None:
        self.explorer.select_experiment(experiment)
        self.selection.set("experiment", experiment)
        self._status_selection.setText(f"ranking → {experiment} › {run_tag}")

    def _on_para_point(self, point) -> None:
        facet = getattr(point, "facet", {}) or {}
        self._status_selection.setText(f"paraconsistent › {point.label}")
        if facet.get("run_tag"):
            self.selection.update(experiment="thesis")

    def _on_cache_failed(self, _key, exc, _tb) -> None:
        from experiment_microscope.core.i18n import t
        self.statusBar().showMessage(t("compute failed: {exc}", exc=exc), 8000)

    def _refresh_session_log(self) -> None:
        adapter = self.repo.adapter("meeting01")
        text = None
        try:
            text = adapter.dashboard_text()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            text = f"session dashboard unavailable: {exc}"
        self.session_log.setPlainText(text or "No meeting01 run detected under results/meeting01/.")

    # -- developer panel (FIXME §37) ---------------------------
    def _probe_shapes(self) -> "dict[str, tuple[int, ...] | None]":
        """Shapes of whatever the pipeline views currently hold. Read-only."""
        out: dict[str, tuple[int, ...] | None] = {
            "raw": None, "transformed": None, "latent": None, "reconstruction": None,
        }
        sig = getattr(self.signal_view, "_last_signal", None)
        if sig is not None:
            out["raw"] = tuple(np.asarray(sig.samples).shape)
        decomp = getattr(self.wavelet_lab, "_decomp", None)
        if decomp is not None:
            out["transformed"] = tuple(np.asarray(decomp.transformed_signal).shape)
        trace = getattr(self.encoding_lab, "_last_trace", None)
        if trace is not None:
            lat = getattr(trace, "latent", None)
            rec = getattr(trace, "reconstruction", None)
            if lat is not None:
                out["latent"] = tuple(np.asarray(lat).shape)
            if rec is not None:
                out["reconstruction"] = tuple(np.asarray(rec).shape)
        return out

    # -- publication export (FIXME §30) -------------------------
    def _export_current_view(self) -> None:
        from experiment_microscope.core.i18n import t

        view = self.tabs.currentWidget()
        if not (hasattr(view, "can_export") and view.can_export()):
            self.statusBar().showMessage(t("current view has nothing to export"), 4000)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, t("Export figure"), "figure.pdf", "Vector/raster (*.pdf *.svg *.png)"
        )
        if not path:
            return
        try:
            written = view.export_figure(path)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(t("export failed: {exc}", exc=exc), 8000)
            return
        self.statusBar().showMessage(t("wrote {written}", written=written), 6000)

    # -- bookmarks (FIXME §36) ----------------------------------
    def _save_bookmark(self, name: str) -> None:
        bm = make_bookmark(
            name,
            self.selection.snapshot(),
            self.explorer.current_path(),
            self._tab_key(self.tabs.currentIndex()),
            bytes(self.saveState()),
        )
        self.bookmarks.store.add(bm)
        self.bookmarks.refresh()
        from experiment_microscope.core.i18n import t
        self.statusBar().showMessage(t("bookmarked: {name}", name=name), 4000)

    def _restore_bookmark(self, bm) -> None:
        if bm.window_state():
            self.restoreState(bm.window_state())
        found = self.explorer.select_path(bm.nav_path) if bm.nav_path else False
        if not found:
            self.selection.restore(bm.selection)
        self._open_tab(bm.active_tab)
        from experiment_microscope.core.i18n import t
        self.statusBar().showMessage(
            t("restored: {name}", name=bm.name)
            + ("" if found else "  " + t("(selection path not found — state only)")),
            5000,
        )

    # -- layout persistence -------------------------------------
    def _clamp_to_screen(self) -> None:
        """Keep the window inside the current monitor's usable area.

        A saved geometry from a bigger display, or the 1400×900 default on a
        smaller one, otherwise leaves the window (and its title bar) partly
        off-screen with no way to grab it.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        margin = 8
        w = min(self.width(), avail.width() - 2 * margin)
        h = min(self.height(), avail.height() - 2 * margin)
        if w < self.width() or h < self.height():
            self.resize(w, h)
        g = self.frameGeometry()
        x = min(max(g.x(), avail.left() + margin), avail.right() - g.width() - margin)
        y = min(max(g.y(), avail.top() + margin), avail.bottom() - g.height() - margin)
        self.move(max(x, avail.left() + margin), max(y, avail.top() + margin))

    def _restore_layout(self) -> bool:
        settings = QSettings(_ORG, _APP)
        geo = settings.value("geometry")
        state = settings.value("windowState")
        if geo is not None:
            self.restoreGeometry(geo)
        if state is not None:
            self.restoreState(state)
        return state is not None

    def _apply_dock_sizes(self, *, force: bool) -> None:
        """Give the side docks a usable width relative to the window.

        Runs unconditionally when there is no saved layout; when ``force`` is
        False it only widens docks that a restored layout left too narrow to
        read (the failure in the screenshot: side panels a few pixels wide).
        """
        win_w = max(self.width(), 800)
        left_w = min(300, max(220, win_w // 5))
        right_w = min(380, max(280, win_w // 4))
        left = getattr(self, "_left_dock", None)
        right_visible = [d for d in getattr(self, "_right_docks", []) if d.isVisible()]
        if left is not None and (force or left.width() < 160):
            self.resizeDocks([left], [left_w], Qt.Orientation.Horizontal)
        if right_visible and (force or min(d.width() for d in right_visible) < 200):
            self.resizeDocks(right_visible, [right_w] * len(right_visible),
                             Qt.Orientation.Horizontal)

    def _reset_layout(self) -> None:
        """View → Reset panel layout — discard the saved arrangement."""
        settings = QSettings(_ORG, _APP)
        settings.remove("windowState")
        settings.remove("geometry")
        for d in getattr(self, "_right_docks", []):
            d.setFloating(False)
            d.setVisible(d is not self._right_docks[-1])  # keep Developer hidden
        for prev, nxt in zip(self._right_docks, self._right_docks[1:]):
            self.tabifyDockWidget(prev, nxt)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)
        self._left_dock.setFloating(False)
        self._left_dock.setVisible(True)
        self._right_docks[0].raise_()
        self._clamp_to_screen()
        self._apply_dock_sizes(force=True)
        from experiment_microscope.core.i18n import t
        self.statusBar().showMessage(t("Panel layout reset."), 4000)

    def closeEvent(self, event) -> None:  # noqa: N802
        settings = QSettings(_ORG, _APP)
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        super().closeEvent(event)
