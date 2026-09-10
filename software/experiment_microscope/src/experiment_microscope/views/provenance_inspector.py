"""Provenance / metadata inspector dock (FIXME §27, §28, §29).

Shows the ``ProvenanceRecord`` for the current selection, each value tagged
with its ``Origin``. A missing value is an em dash, never a zero.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from experiment_microscope.data.adapters import ProvenanceRecord, TreeNode
from experiment_microscope.data.repository import DataRepository


class ProvenanceInspector(QTreeWidget):
    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        from experiment_microscope.core.i18n import t as _t
        self.setHeaderLabels([_t("Field"), _t("Value"), _t("Origin")])
        self.setColumnWidth(0, 160)
        self.setColumnWidth(1, 220)
        self._show_placeholder("Select an object in the explorer.")

    def _show_placeholder(self, text: str) -> None:
        self.clear()
        item = QTreeWidgetItem([text, "", ""])
        item.setDisabled(True)
        self.addTopLevelItem(item)

    def show_node(self, node: TreeNode, adapter_key: str) -> None:
        adapter = self.repo.adapter(adapter_key)
        try:
            record = adapter.load_provenance(node)
        except NotImplementedError:
            self._show_placeholder("No provenance for this object.")
            return
        except Exception as exc:  # noqa: BLE001
            self._show_placeholder(f"Provenance unavailable: {exc}")
            return
        self._render(record)

    def _render(self, record: ProvenanceRecord) -> None:
        self.clear()
        for section, mapping in (
            ("Source", record.source),
            ("Processing", record.processing),
            ("Model", record.model),
            ("Artifact", record.artifact),
        ):
            head = QTreeWidgetItem([section, "", ""])
            self.addTopLevelItem(head)
            head.setExpanded(True)
            for field, value in mapping.items():
                origin = "" if value.is_missing else value.origin.value
                child = QTreeWidgetItem([field, value.display(), origin])
                if value.note:
                    child.setToolTip(1, value.note)
                head.addChild(child)
