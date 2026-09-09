"""Cross-run ranking / model-comparison table (FIXME §21, §12).

Every persisted paraconsistent score, from every experiment, in one sortable and
filterable table. This is the "which feature set / model actually ranked well"
screen: the ranking metric is ``d_penalized`` (ascending — lower is closer to
the Truth vertex with the contradiction penalty applied).

Only columns that have a value for a row are meaningful; a missing cell shows
``—`` (never 0). Double-clicking a row selects that experiment so the explorer
and the other views follow.
"""

from __future__ import annotations

from experiment_microscope.views._help import HelpBox

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.data.repository import DataRepository

_COLS = ("experiment", "run_tag", "feature set", "modality/enc", "seed",
         "alpha", "beta", "G1", "G2", "d_truth", "d_penalized")


_HELP = """
<b>What this shows.</b> Every persisted <b>paraconsistent</b> score from every
experiment in one sortable table: experiment, run, feature set, modality /
encoding, seed, and then α, β, G1, G2, D_truth, <b>D_penalized</b>.
<br><br>
Default sort is D_penalized ascending — <b>smaller is better</b> (it is the
distance to "certainly true, no contradiction", with a contradiction penalty).
Missing values show as "—" and sort last, never as 0. Type in the filter box to
narrow by any text. Double-click a row to jump to that experiment.
"""


class _NumItem(QTableWidgetItem):
    """Numeric cell that keeps its own sort key (QTableWidgetItem folds
    EditRole into DisplayRole, so a hidden numeric role is not an option)."""

    def __init__(self, value) -> None:
        missing = value is None or (isinstance(value, float) and value != value)
        super().__init__("—" if missing else f"{float(value):.6g}")
        self._key = float("inf") if missing else float(value)
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other: "QTableWidgetItem") -> bool:  # type: ignore[override]
        return self._key < getattr(other, "_key", float("inf"))


def _num_item(value) -> QTableWidgetItem:
    return _NumItem(value)


def _mag(v):
    return None if (v is None or getattr(v, "is_missing", False)) else float(v.magnitude)


class RankingView(QWidget):
    #: (experiment_key, run_tag)
    run_activated = Signal(str, str)

    def __init__(self, repo: DataRepository, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.repo = repo
        self._rows: list[tuple] = []

        root = QVBoxLayout(self)

        root.addWidget(HelpBox('Ranking', _HELP))
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("filter (substring over experiment / run_tag / feature set)…")
        self._filter.textChanged.connect(self._apply_filter)
        root.addWidget(self._filter)

        self._table = QTableWidget(0, len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.itemDoubleClicked.connect(self._on_double_click)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        root.addWidget(self._table, 1)
        self.refresh()

    def show_node(self, node, adapter_key: str) -> None:  # uniform view API
        """Pipeline-wide; a selection does not change the table."""

    def refresh(self) -> None:
        points = self.repo.all_paraconsistent_points()
        self._rows = []
        for p in points:
            f = p.facet or {}
            experiment = "thesis" if "phase" in f else (
                "paraconsistent_ga" if "encoding" in f or "depth" in f else "unknown"
            )
            seed = f.get("seed")
            seed = _mag(seed) if hasattr(seed, "magnitude") else seed
            self._rows.append((
                experiment,
                str(f.get("run_tag", "")),
                str(f.get("feature_set") or ""),
                str(f.get("modality") or f.get("encoding") or ""),
                seed,
                _mag(p.alpha), _mag(p.beta), _mag(p.g1), _mag(p.g2),
                _mag(p.d_truth), _mag(p.d_penalized),
            ))
        self._populate()

    def _populate(self) -> None:
        needle = self._filter.text().strip().lower()
        rows = [r for r in self._rows
                if not needle or needle in f"{r[0]} {r[1]} {r[2]}".lower()]
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for c in (0, 1, 2, 3):
                it = QTableWidgetItem(str(r[c]))
                self._table.setItem(i, c, it)
            self._table.setItem(i, 4, _num_item(r[4]))
            for c in range(5, 11):
                self._table.setItem(i, c, _num_item(r[c]))
        self._table.setSortingEnabled(True)
        self._table.sortItems(_COLS.index("d_penalized"), Qt.SortOrder.AscendingOrder)

    def _apply_filter(self) -> None:
        self._populate()

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        exp = self._table.item(row, 0).text()
        tag = self._table.item(row, 1).text()
        if exp in ("thesis", "paraconsistent_ga"):
            self.run_activated.emit(exp, tag)
