"""Global search over the Data Explorer (FIXME §35).

One box that matches every whitespace-separated token against a node's whole
path plus its metadata (kind, modality, strategy, seed, fold, wavelet/scale
encoded in the run tag, …). A hit is a navigation path; activating it drives
``ExplorerTree.select_path`` so every view follows.

The catalog is built lazily on first use (it force-expands the tree to a bounded
depth) and rebuilt on demand.
"""

from __future__ import annotations

from experiment_microscope.core.i18n import t as _t
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

_MAX_HITS = 50


class SearchBar(QWidget):
    #: navigation path (list of tree labels) chosen by the user
    path_activated = Signal(list)

    def __init__(self, explorer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._explorer = explorer
        self._catalog: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        self._edit = QLineEdit()
        self._edit.setPlaceholderText(_t('search — e.g. "daub10 lfcc eeg" or "fold 0 fsdd"'))
        self._edit.setClearButtonEnabled(True)
        self._edit.textChanged.connect(self._on_text)
        self._edit.returnPressed.connect(self._activate_current)
        root.addWidget(self._edit)
        self._results = QListWidget()
        self._results.itemActivated.connect(self._activate_item)
        self._results.itemDoubleClicked.connect(self._activate_item)
        self._results.setVisible(False)
        root.addWidget(self._results)
        self._hint = QLabel("")
        self._hint.setStyleSheet("color: gray;")
        root.addWidget(self._hint)

    def refresh_catalog(self) -> None:
        self._catalog = self._explorer.build_catalog()

    # -- query ------------------------------------------------
    def _on_text(self, text: str) -> None:
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            self._results.clear()
            self._results.setVisible(False)
            self._hint.setText("")
            return
        if not self._catalog:
            self.refresh_catalog()
        hits = [e for e in self._catalog if all(tok in e["text"] for tok in tokens)]
        self._results.clear()
        for e in hits[:_MAX_HITS]:
            it = QListWidgetItem(f"{'  ›  '.join(e['path'])}   ({e['kind']})")
            it.setData(Qt.ItemDataRole.UserRole, e["path"])
            self._results.addItem(it)
        self._results.setVisible(bool(hits))
        shown = min(len(hits), _MAX_HITS)
        self._hint.setText(
            "" if not hits else
            _t("{shown} of {total} match(es)", shown=shown, total=len(hits))
            + (_t(" — refine to see more") if len(hits) > shown else "")
        )

    def _activate_current(self) -> None:
        if self._results.count():
            self._activate_item(self._results.item(self._results.currentRow())
                                or self._results.item(0))

    def _activate_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.path_activated.emit(list(path))
