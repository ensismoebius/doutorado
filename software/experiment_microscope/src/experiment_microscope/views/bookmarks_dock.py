"""Bookmarks dock (FIXME §36)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.bookmarks import Bookmark, BookmarkStore

_ORG, _APP = "doutorado", "experiment_microscope"


class BookmarksDock(QWidget):
    #: user asked to save the current state under this name
    save_requested = Signal(str)
    #: user asked to restore this bookmark
    restore_requested = Signal(object)  # Bookmark

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = BookmarkStore(_ORG, _APP)
        layout = QVBoxLayout(self)
        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)
        buttons = QHBoxLayout()
        add = QPushButton("Add current…")
        add.clicked.connect(self._on_add)
        rm = QPushButton("Remove")
        rm.clicked.connect(self._on_remove)
        buttons.addWidget(add)
        buttons.addWidget(rm)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self) -> None:
        self._list.clear()
        for b in self.store.load():
            item = QListWidgetItem(b.name)
            item.setToolTip(
                f"{b.selection.get('experiment') or '?'} · tab {b.active_tab or '?'}"
            )
            item.setData(Qt.ItemDataRole.UserRole, b)
            self._list.addItem(item)

    def _on_add(self) -> None:
        name, ok = QInputDialog.getText(self, "Add bookmark", "Name:")
        if ok and name.strip():
            self.save_requested.emit(name.strip())

    def _on_remove(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        b: Bookmark = item.data(Qt.ItemDataRole.UserRole)
        self.store.remove(b.name)
        self.refresh()

    def _on_double_click(self, item: QListWidgetItem) -> None:
        b = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(b, Bookmark):
            self.restore_requested.emit(b)
