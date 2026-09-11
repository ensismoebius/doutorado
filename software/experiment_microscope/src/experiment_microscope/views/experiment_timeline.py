"""Meeting01 experiment timeline (FIXME §45).

    session
     ├ fsdd / fold 0
     │   ├ snn-ae_direct_seed42_run1        best val 0.0123 @ epoch 7  (done)
     │   │   ├ epoch 1   train 0.09  val 0.08
     │   │   └ …
     │   └ lstm-ae_direct_seed42_run1       running
     └ fsdd / fold 1
     …

Built from the same ``*_events.jsonl`` stream the CLI monitor reads
(``monitor._drain`` → ``SessionState``). Selecting a config emits
``config_activated(dataset, fold, config_id)`` so the rest of the app can jump
to that fold. Empty (with a reason) until a meeting01 run has written events —
the expected state right after the results purge.
"""

from __future__ import annotations

from experiment_microscope.views._help import HelpBox
from experiment_microscope.core.i18n import t as _t

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: cycled for each config in a multi-fold comparison
_COMPARE_COLORS = [
    (120, 170, 255), (255, 170, 90), (120, 220, 150), (230, 120, 200),
    (240, 210, 90), (140, 200, 230), (200, 140, 240), (230, 90, 90),
]

from experiment_microscope.views._pg import PG_OK, missing_widget, pg

_ROLE = Qt.ItemDataRole.UserRole


_HELP = """
<b>What this shows.</b> The meeting01 run as a tree: session → dataset / fold →
config → epoch, built live from the structured event log.
<br><br>
Selecting a config plots its learning curve below: <b>train loss</b> and
<b>validation loss</b> per epoch (loss = reconstruction error the optimiser
minimises), a dashed line at the best epoch, epoch <b>duration</b> on the right
axis, and the learning rate in the title. A rising validation curve while train
keeps falling is the classic overfitting shape — but the tool only shows it, it
does not label it.
<br><br>
<b>Compare folds.</b> Ctrl/Shift-click several configs (or a whole fold row,
which selects every config under it) to overlay their curves — one colour per
config, <b>solid = validation loss</b>, <b>dotted = train loss</b> — so you can
see whether the network behaves the same way across folds or diverges on one.
<br><br>
Empty until a LOSO run has written <code>results/meeting01/*_events.jsonl</code>.
"""


class ExperimentTimeline(QWidget):
    #: (dataset, fold, config_id)
    config_activated = Signal(str, int, str)

    def __init__(self, repo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        root = QVBoxLayout(self)
        root.addWidget(HelpBox('Experiment timeline', _HELP))
        self._status = QLabel(_t("No meeting01 event stream yet."))
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        split = QSplitter(Qt.Orientation.Vertical)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels([_t("node"), _t("detail")])
        self._tree.setColumnWidth(0, 320)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # single click (or arrow-key move) plots — Ctrl/Shift-click adds more
        # configs to the comparison (FIXME §45 — "how does it behave across folds").
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        # double-click / Enter still jumps the rest of the app to that fold.
        self._tree.itemActivated.connect(self._on_activated)
        split.addWidget(self._tree)

        # training curve for the selected config (FIXME §46)
        self._configs: dict[str, object] = {}
        self._epoch_ms_vb = None
        if PG_OK:
            self._curve = pg.PlotWidget()
            self._curve.addLegend()
            self._curve.setLabel("bottom", _t("epoch"))
            self._curve.setLabel("left", _t("loss"))
            self._curve.showGrid(x=True, y=True, alpha=0.3)
            from experiment_microscope.views._plotinfo import HoverReadout
            self._hover = HoverReadout(self._curve, x_label="epoch")
            split.addWidget(self._curve)
        else:
            self._curve = None
            split.addWidget(missing_widget("training curve"))
        split.setSizes([420, 200])
        root.addWidget(split, 1)

    def show_node(self, node, adapter_key: str) -> None:
        if adapter_key == "meeting01":
            self.refresh()

    # -- build ------------------------------------------------
    def refresh(self) -> None:
        self._tree.clear()
        adapter = self.repo.adapter("meeting01")
        state = None
        try:
            state = adapter.session_state()
        except Exception as exc:  # noqa: BLE001
            self._status.setText(_t("event stream unreadable: {exc}", exc=exc))
            return
        if state is None or not getattr(state, "configs", None):
            self._status.setText(_t(
                "No meeting01 event stream under results/meeting01/ "
                "(*_events.jsonl). Run a LOSO fold to populate this."
            ))
            return

        sess = getattr(state, "session", {}) or {}
        commit = sess.get("git_commit", "?")
        seed = sess.get("seed", "?")
        self._status.setText(_t(
            "session — git {git}, seed {seed}, {nc} config(s), {nf} fold(s)",
            git=str(commit)[:10], seed=seed, nc=len(state.configs),
            nf=len(getattr(state, 'folds_seen', ()))))

        groups: dict[tuple[str, int], list] = {}
        self._configs = {}
        for cfg in state.configs.values():
            groups.setdefault((cfg.dataset or "?", int(cfg.fold)), []).append(cfg)
            self._configs[cfg.config_id] = cfg

        for (ds, fold), cfgs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            fold_item = QTreeWidgetItem([f"{ds} / fold {fold}",
                                        _t("{n} config(s)", n=len(cfgs))])
            self._tree.addTopLevelItem(fold_item)
            fold_item.setExpanded(True)
            for cfg in sorted(cfgs, key=lambda c: c.config_id):
                detail = cfg.status
                if cfg.best_val is not None:
                    detail += _t("  ·  best val {v}", v=f"{cfg.best_val:.6g}")
                    if cfg.best_epoch is not None:
                        detail += _t(" @ epoch {e}", e=cfg.best_epoch)
                c_item = QTreeWidgetItem([cfg.config_id, detail])
                c_item.setData(0, _ROLE, (ds, fold, cfg.config_id))
                fold_item.addChild(c_item)
                for ep, tr, val in cfg.epochs:
                    txt = _t("train {tr}   val {val}", tr=_fmt(tr), val=_fmt(val))
                    c_item.addChild(QTreeWidgetItem([_t("epoch {e}", e=ep), txt]))

    def _item_payload(self, item: QTreeWidgetItem):
        payload = item.data(0, _ROLE)
        if payload is None and item.parent() is not None:
            payload = item.parent().data(0, _ROLE)  # an epoch row → its config
        return payload

    def _on_activated(self, item: QTreeWidgetItem | None, _col: int) -> None:
        """Double-click / Enter — jump the rest of the app to this fold."""
        if item is None:
            return
        payload = self._item_payload(item)
        if payload:
            ds, fold, cfg_id = payload
            self.config_activated.emit(ds, int(fold), cfg_id)

    def _selected_config_ids(self) -> list[str]:
        """Config ids implied by the current selection, in encounter order,
        each once. Selecting a fold row stands in for every config under it —
        the one-click way to compare a whole fold."""
        ids: list[str] = []
        seen: set[str] = set()
        for item in self._tree.selectedItems():
            payload = item.data(0, _ROLE)
            if payload is not None:
                cfg_id = payload[2]
                if cfg_id not in seen:
                    seen.add(cfg_id)
                    ids.append(cfg_id)
            elif item.parent() is None:  # a fold row — every child config
                for i in range(item.childCount()):
                    child_payload = item.child(i).data(0, _ROLE)
                    if child_payload is not None and child_payload[2] not in seen:
                        seen.add(child_payload[2])
                        ids.append(child_payload[2])
        return ids

    def _on_selection_changed(self) -> None:
        ids = self._selected_config_ids()
        if not ids:
            return
        if len(ids) == 1:
            self._plot_curve(ids[0])
        else:
            self._plot_compare(ids)

    def _clear_legend(self) -> None:
        legend = getattr(self._curve.getPlotItem(), "legend", None)
        if legend is not None:
            legend.clear()

    def _plot_curve(self, config_id: str) -> None:
        if self._curve is None:
            return
        self._curve.clear()
        self._clear_legend()
        cfg = self._configs.get(config_id)
        epochs = list(getattr(cfg, "epochs", []) or [])
        if not epochs:
            self._curve.setTitle(_t("{cfg} — no epoch data", cfg=config_id))
            return
        xs = [e[0] for e in epochs]
        tr = [e[1] if e[1] is not None else float("nan") for e in epochs]
        val = [e[2] if e[2] is not None else float("nan") for e in epochs]
        self._curve.plot(xs, tr, pen=pg.mkPen((120, 170, 255), width=2), name=_t("train"))
        self._curve.plot(xs, val, pen=pg.mkPen((255, 170, 90), width=2), name=_t("val"))
        best = getattr(cfg, "best_epoch", None)
        if best is not None:
            self._curve.addLine(x=best, pen=pg.mkPen((90, 200, 120), style=Qt.PenStyle.DashLine))
        from experiment_microscope.views._plotinfo import autofit, set_source
        set_source(self._curve, "results/meeting01/*_events.jsonl (epoch_end events)")
        if getattr(self, "_hover", None) is not None:
            self._hover.reattach()
        autofit(self._curve)  # re-fit every regeneration so all epochs stay in frame

        # epoch duration on a linked right-hand axis (FIXME §46)
        self._plot_epoch_ms(cfg, xs)
        lr = getattr(cfg, "lr", None)
        title = f"{config_id}  ·  {getattr(cfg, 'status', '')}"
        if lr is not None:
            title += _t("  ·  lr {lr}", lr=f"{lr:g}")
        avg_ms = getattr(cfg, "avg_epoch_ms", None)
        if callable(avg_ms):  # monitor.ConfigState exposes it as a method, not a property
            try:
                avg_ms = avg_ms()
            except Exception:  # noqa: BLE001
                avg_ms = None
        if avg_ms:
            title += _t("  ·  ~{s}s/epoch", s=f"{avg_ms / 1000:.1f}")
        self._curve.setTitle(title)

    def _plot_compare(self, config_ids: list[str]) -> None:
        """Overlay several configs' curves (FIXME §45) — one colour per config,
        solid = validation loss, dotted = train loss. Answers "does the network
        behave the same way across folds, or does one diverge?"."""
        if self._curve is None:
            return
        self._curve.clear()
        self._clear_legend()
        self._hide_epoch_ms_axis()
        plotted = 0
        for i, config_id in enumerate(config_ids):
            cfg = self._configs.get(config_id)
            epochs = list(getattr(cfg, "epochs", []) or [])
            if not epochs:
                continue
            xs = [e[0] for e in epochs]
            tr = [e[1] if e[1] is not None else float("nan") for e in epochs]
            val = [e[2] if e[2] is not None else float("nan") for e in epochs]
            color = _COMPARE_COLORS[i % len(_COMPARE_COLORS)]
            self._curve.plot(xs, val, pen=pg.mkPen(color, width=2), name=config_id)
            self._curve.plot(xs, tr, pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DotLine))
            plotted += 1
        from experiment_microscope.views._plotinfo import autofit, set_source
        set_source(self._curve, "results/meeting01/*_events.jsonl (epoch_end events)")
        if getattr(self, "_hover", None) is not None:
            self._hover.reattach()
        autofit(self._curve)
        self._curve.setTitle(_t(
            "comparing {n} configs — solid = val loss, dotted = train loss",
            n=plotted))

    def _hide_epoch_ms_axis(self) -> None:
        if self._epoch_ms_vb is not None:
            self._epoch_ms_vb.clear()
        self._curve.getPlotItem().hideAxis("right")

    def _plot_epoch_ms(self, cfg, xs: list[int]) -> None:
        ems = list(getattr(cfg, "_epoch_ms", []) or [])
        if not ems or self._curve is None:
            self._epoch_ms_vb = None
            return
        pi = self._curve.getPlotItem()
        pi.showAxis("right")  # compare mode may have hidden it
        if getattr(self, "_epoch_ms_vb", None) is None:
            self._epoch_ms_vb = pg.ViewBox()
            pi.scene().addItem(self._epoch_ms_vb)
            pi.getAxis("right").linkToView(self._epoch_ms_vb)
            self._epoch_ms_vb.setXLink(pi)
            pi.showAxis("right")
            pi.getAxis("right").setLabel(_t("epoch duration (s)"), color="#b0b0b0")
            pi.vb.sigResized.connect(
                lambda: self._epoch_ms_vb.setGeometry(pi.vb.sceneBoundingRect())
            )
        self._epoch_ms_vb.clear()
        n = min(len(ems), len(xs))
        curve = pg.PlotDataItem(xs[:n], [v / 1000.0 for v in ems[:n]],
                                pen=pg.mkPen((160, 120, 200), width=1, style=Qt.PenStyle.DotLine))
        self._epoch_ms_vb.addItem(curve)
        self._epoch_ms_vb.setGeometry(pi.vb.sceneBoundingRect())
        self._epoch_ms_vb.enableAutoRange(y=True)
        self._epoch_ms_vb.autoRange(padding=0.08)  # fit the duration curve every regen


def _fmt(x) -> str:
    return "—" if x is None else f"{float(x):.6g}"
