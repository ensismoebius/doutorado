"""Didactic helpers shared by every view (FIXME §29, §33, §55).

* ``HelpBox`` — a collapsible "How to read this" strip that sits at the top of a
  view. Collapsed by default so it never crowds the data, one click to open.
* ``metric_table`` / ``fill_metric_table`` — a 3-column table (name · value ·
  what it means) so no number is shown without a description.
* ``label_plot`` — force a pyqtgraph plot to carry axis labels + a legend.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from experiment_microscope.core.glossary import describe
from experiment_microscope.core.i18n import t


class HelpBox(QWidget):
    """A one-click-to-expand explanation strip.

    ``body`` is rich text (HTML). Keep it to: what this view shows, what each
    axis / number means (abbreviations spelled out), and what to do next.
    """

    def __init__(self, title: str, body: str, parent: QWidget | None = None,
                 start_open: bool = False) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 4)
        lay.setSpacing(2)

        self._btn = QToolButton()
        self._btn.setText(f"  {t('How to read this')} — {t(title)}")
        self._btn.setCheckable(True)
        self._btn.setChecked(start_open)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if start_open else Qt.ArrowType.RightArrow)
        self._btn.setAutoRaise(True)
        self._btn.toggled.connect(self._on_toggle)
        lay.addWidget(self._btn)

        self._body = QLabel(t(body))
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.TextFormat.RichText)
        self._body.setFrameShape(QFrame.Shape.StyledPanel)
        self._body.setMargin(8)
        self._body.setVisible(start_open)
        self._body.setOpenExternalLinks(False)
        lay.addWidget(self._body)

    def _on_toggle(self, on: bool) -> None:
        self._body.setVisible(on)
        self._btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def open(self) -> None:
        self._btn.setChecked(True)

    def set_body(self, body: str) -> None:
        self._body.setText(body)


def metric_table() -> QTableWidget:
    """A 3-column read-only table: metric · value · what it means."""
    from experiment_microscope.core.i18n import t as _t
    t = QTableWidget(0, 3)
    t.setHorizontalHeaderLabels([_t("quantity"), _t("value"), _t("what it means")])
    t.horizontalHeader().setStretchLastSection(True)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setWordWrap(True)
    t.verticalHeader().setVisible(False)
    return t


def fill_metric_table(table: QTableWidget, rows: list[tuple[str, str, str]]) -> None:
    """``rows`` = list of (name, value, meaning). An empty meaning is looked up
    in the glossary by ``name``."""
    table.setRowCount(len(rows))
    for r, (name, value, meaning) in enumerate(rows):
        table.setItem(r, 0, QTableWidgetItem(name))
        table.setItem(r, 1, QTableWidgetItem(value))
        table.setItem(r, 2, QTableWidgetItem(meaning or describe(name) or "—"))
    table.resizeRowsToContents()


def label_plot(plot, *, bottom: str, left: str, title: str | None = None,
               legend: bool = True) -> None:
    """Give a pyqtgraph PlotItem/PlotWidget axis labels (with units in the text),
    a grid and a legend. Call once per plot, before adding named curves."""
    try:
        pi = plot.getPlotItem() if hasattr(plot, "getPlotItem") else plot
        pi.setLabel("bottom", bottom)
        pi.setLabel("left", left)
        if title is not None:
            pi.setTitle(title)
        pi.showGrid(x=True, y=True, alpha=0.2)
        if legend and pi.legend is None:
            pi.addLegend(offset=(-10, 10))
    except Exception:  # noqa: BLE001 - never let a label call break a view
        pass
