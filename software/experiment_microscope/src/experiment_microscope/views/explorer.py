"""Data Explorer dock (FIXME §7).

Lazy hierarchical tree over every adapter. Selecting a node writes the coarse
fields of the shared ``SelectionState``; downstream views react to that, not to
this widget.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository

_NODE_ROLE = Qt.ItemDataRole.UserRole
_ADAPTER_ROLE = Qt.ItemDataRole.UserRole + 1
_LOADED_ROLE = Qt.ItemDataRole.UserRole + 2


class ExplorerTree(QTreeWidget):
    node_selected = Signal(object, str)  # (TreeNode, adapter_key)

    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self.setHeaderLabels(["Object", "Info"])
        self.setColumnWidth(0, 260)
        self.itemExpanded.connect(self._on_expanded)
        self.currentItemChanged.connect(self._on_current)
        self._populate_roots()

    # -- build ------------------------------------------------------
    def _populate_roots(self) -> None:
        self.clear()
        for key, nodes in self.repo.root_nodes():
            for node in nodes:
                self._add_item(self.invisibleRootItem(), node, key)

    def _add_item(self, parent: QTreeWidgetItem, node: TreeNode, adapter_key: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [node.label, self._info(node)])
        item.setData(0, _NODE_ROLE, node)
        item.setData(0, _ADAPTER_ROLE, adapter_key)
        item.setData(0, _LOADED_ROLE, False)
        if node.has_children:
            # placeholder so the expand arrow shows before we load
            QTreeWidgetItem(item, ["…", ""])
        return item

    @staticmethod
    def _info(node: TreeNode) -> str:
        md = node.metadata or {}
        for key in ("status", "modality", "strategy", "on_disk", "has_results", "pareto_front_size"):
            if key in md and md[key] is not None:
                return f"{key}={md[key]}"
        return node.kind

    # -- lazy expand ---------------------------------------------
    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        if item.data(0, _LOADED_ROLE):
            return
        item.takeChildren()
        node: TreeNode = item.data(0, _NODE_ROLE)
        key: str = item.data(0, _ADAPTER_ROLE)
        adapter = self.repo.adapter(key)
        try:
            children = adapter.children(node)
        except Exception as exc:  # noqa: BLE001
            err = QTreeWidgetItem(item, ["<error>", str(exc)])
            err.setDisabled(True)
            item.setData(0, _LOADED_ROLE, True)
            return
        for child in children:
            self._add_item(item, child, key)
        item.setData(0, _LOADED_ROLE, True)

    def _on_current(self, current: QTreeWidgetItem | None, _prev) -> None:
        if current is None:
            return
        node = current.data(0, _NODE_ROLE)
        key = current.data(0, _ADAPTER_ROLE)
        if node is not None and key:
            self.node_selected.emit(node, key)

    def select_experiment(self, key: str) -> None:
        root = self.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            if item.data(0, _ADAPTER_ROLE) == key:
                self.setCurrentItem(item)
                item.setExpanded(True)
                return

    def current_path(self) -> list[str]:
        """Labels from a top-level item down to the current selection."""
        item = self.currentItem()
        path: list[str] = []
        while item is not None:
            path.insert(0, item.text(0))
            item = item.parent()
        return path

    def build_catalog(self, max_depth: int = 4, max_children: int = 1500) -> list[dict]:
        """Force-expand the tree to ``max_depth`` and return searchable entries
        (FIXME §35). Each entry: ``{path: [labels], text: str, kind: str}``.

        Nodes with more than ``max_children`` children (window / sample lists,
        capped at 200 by the adapters anyway) are recorded but not descended —
        searching individual windows is out of scope for a global catalog."""
        out: list[dict] = []
        root = self.invisibleRootItem()

        def walk(item: QTreeWidgetItem, path: list[str], depth: int) -> None:
            node: TreeNode = item.data(0, _NODE_ROLE)
            if node is None:
                return
            md = node.metadata or {}
            text = " ".join(str(x) for x in (
                *path, node.label, node.kind, self._info(node),
                *(f"{k}={v}" for k, v in md.items() if v is not None),
            )).lower()
            out.append({"path": list(path), "text": text, "kind": node.kind})
            if depth >= max_depth or not node.has_children:
                return
            item.setExpanded(True)
            self._on_expanded(item)
            if item.childCount() > max_children:
                return
            for i in range(item.childCount()):
                child = item.child(i)
                walk(child, path + [child.text(0)], depth + 1)

        for i in range(root.childCount()):
            top = root.child(i)
            walk(top, [top.text(0)], 1)
        return out

    def select_path(self, labels: list[str]) -> bool:
        """Expand + select the item at ``labels`` (top-level label first).

        Returns True if the full path was found. Fires ``node_selected`` for
        the final item, which repopulates every view (FIXME §36 restore)."""
        if not labels:
            return False
        parent = self.invisibleRootItem()
        item = None
        for label in labels:
            item = next(
                (parent.child(i) for i in range(parent.childCount())
                 if parent.child(i).text(0) == label),
                None,
            )
            if item is None:
                return False
            item.setExpanded(True)  # triggers lazy _on_expanded
            self._on_expanded(item)
            parent = item
        if item is not None:
            self.setCurrentItem(item)
            return True
        return False
