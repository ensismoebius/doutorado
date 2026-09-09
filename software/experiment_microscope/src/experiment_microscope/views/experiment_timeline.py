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
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

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
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["node", "detail"])
        self._tree.setColumnWidth(0, 320)
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.currentItemChanged.connect(lambda cur, _prev: self._on_activated(cur, 0))
        root.addWidget(self._tree, 1)

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
        for cfg in state.configs.values():
            groups.setdefault((cfg.dataset or "?", int(cfg.fold)), []).append(cfg)

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
            self.config_activated.emit(ds, int(fold), cfg_id)


def _fmt(x) -> str:
    return "—" if x is None else f"{float(x):.6g}"
