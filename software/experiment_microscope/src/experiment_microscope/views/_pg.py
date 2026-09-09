"""Guarded pyqtgraph import.

pyqtgraph is a hard dependency of the app (pyproject), but the test venv or a
minimal checkout may not have it. Views import ``pg`` / ``PG_OK`` from here and
fall back to a plain message widget when it is absent, so the workspace still
opens.
"""

from __future__ import annotations

try:  # pragma: no cover - trivial import guard
    import pyqtgraph as pg  # type: ignore

    PG_OK = True
except Exception:  # noqa: BLE001
    pg = None  # type: ignore
    PG_OK = False


def missing_widget(what: str):
    from PySide6.QtWidgets import QLabel

    label = QLabel(f"{what} needs pyqtgraph — `pip install pyqtgraph` (it is in pyproject.toml).")
    label.setWordWrap(True)
    label.setMargin(12)
    return label
