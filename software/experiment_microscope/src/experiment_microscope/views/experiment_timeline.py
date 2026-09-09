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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.views._pg import PG_OK, missing_widget, pg

_ROLE = Qt.ItemDataRole.UserRole


class ExperimentTimeline(QWidget):
    #: (dataset, fold, config_id)
    config_activated = Signal(str, int, str)

    def __init__(self, repo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        root = QVBoxLayout(self)
        self._status = QLabel("No meeting01 event stream yet.")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        split = QSplitter(Qt.Orientation.Vertical)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["node", "detail"])
        self._tree.setColumnWidth(0, 320)
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.currentItemChanged.connect(lambda cur, _prev: self._on_activated(cur, 0))
        split.addWidget(self._tree)

        # training curve for the selected config (FIXME §46)
        self._configs: dict[str, object] = {}
        if PG_OK:
            self._curve = pg.PlotWidget()
            self._curve.addLegend()
            self._curve.setLabel("bottom", "epoch")
            self._curve.setLabel("left", "loss")
            self._curve.showGrid(x=True, y=True, alpha=0.3)
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
            self._status.setText(f"event stream unreadable: {exc}")
            return
        if state is None or not getattr(state, "configs", None):
            self._status.setText(
                "No meeting01 event stream under results/meeting01/ "
                "(*_events.jsonl). Run a LOSO fold to populate this."
            )
            return

        sess = getattr(state, "session", {}) or {}
        commit = sess.get("git_commit", "?")
        seed = sess.get("seed", "?")
        self._status.setText(
            f"session — git {str(commit)[:10]}, seed {seed}, "
            f"{len(state.configs)} config(s), {len(getattr(state, 'folds_seen', ()))} fold(s)"
        )

        groups: dict[tuple[str, int], list] = {}
        self._configs = {}
        for cfg in state.configs.values():
            groups.setdefault((cfg.dataset or "?", int(cfg.fold)), []).append(cfg)
            self._configs[cfg.config_id] = cfg

        for (ds, fold), cfgs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            fold_item = QTreeWidgetItem([f"{ds} / fold {fold}", f"{len(cfgs)} config(s)"])
            self._tree.addTopLevelItem(fold_item)
            fold_item.setExpanded(True)
            for cfg in sorted(cfgs, key=lambda c: c.config_id):
                detail = cfg.status
                if cfg.best_val is not None:
                    detail += f"  ·  best val {cfg.best_val:.6g}"
                    if cfg.best_epoch is not None:
                        detail += f" @ epoch {cfg.best_epoch}"
                c_item = QTreeWidgetItem([cfg.config_id, detail])
                c_item.setData(0, _ROLE, (ds, fold, cfg.config_id))
                fold_item.addChild(c_item)
                for ep, tr, val in cfg.epochs:
                    txt = f"train {_fmt(tr)}   val {_fmt(val)}"
                    c_item.addChild(QTreeWidgetItem([f"epoch {ep}", txt]))

    def _on_activated(self, item: QTreeWidgetItem | None, _col: int) -> None:
        if item is None:
            return
        payload = item.data(0, _ROLE)
        if payload is None and item.parent() is not None:
            payload = item.parent().data(0, _ROLE)  # an epoch row → its config
        if payload:
            ds, fold, cfg_id = payload
            self._plot_curve(cfg_id)
            self.config_activated.emit(ds, int(fold), cfg_id)

    def _plot_curve(self, config_id: str) -> None:
        if self._curve is None:
            return
        self._curve.clear()
        cfg = self._configs.get(config_id)
        epochs = list(getattr(cfg, "epochs", []) or [])
        if not epochs:
            self._curve.setTitle(f"{config_id} — no epoch data")
            return
        xs = [e[0] for e in epochs]
        tr = [e[1] if e[1] is not None else float("nan") for e in epochs]
        val = [e[2] if e[2] is not None else float("nan") for e in epochs]
        self._curve.plot(xs, tr, pen=pg.mkPen((120, 170, 255), width=2), name="train")
        self._curve.plot(xs, val, pen=pg.mkPen((255, 170, 90), width=2), name="val")
        best = getattr(cfg, "best_epoch", None)
        if best is not None:
            self._curve.addLine(x=best, pen=pg.mkPen((90, 200, 120), style=Qt.PenStyle.DashLine))
        self._curve.setTitle(f"{config_id}  ·  {getattr(cfg, 'status', '')}")


def _fmt(x) -> str:
    return "—" if x is None else f"{float(x):.6g}"
