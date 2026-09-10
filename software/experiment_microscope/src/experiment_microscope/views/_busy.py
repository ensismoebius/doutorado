"""A small busy indicator for the status bar (didactic redesign).

Any time the app is loading a dataset, running the network, computing a wavelet
or projecting the latent space, a moving progress bar + a plain-language label
appear at the right of the status bar. Refcounted: nested / overlapping tasks
keep it visible until the last one ends.
"""

from __future__ import annotations

from contextlib import contextmanager

from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget

from experiment_microscope.core.i18n import t


class BusyIndicator(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._depth = 0
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._label = QLabel("")
        self._label.setStyleSheet("color:#bbb;")
        self._bar = QProgressBar()
        self._bar.setRange(0, 0)            # 0,0 = indeterminate "working" animation
        self._bar.setFixedWidth(120)
        self._bar.setTextVisible(False)
        row.addWidget(self._label)
        row.addWidget(self._bar)
        self.setVisible(False)

    def begin(self, message: str = "") -> None:
        self._depth += 1
        self._label.setText(t(message) if message else t("Working…"))
        self.setVisible(True)
        self.repaint()

    def end(self) -> None:
        self._depth = max(0, self._depth - 1)
        if self._depth == 0:
            self.setVisible(False)

    def on_count(self, n: int) -> None:
        """Slot for ``TransformationCache.busy_changed`` — show while jobs run."""
        if n > 0 and not self.isVisible():
            self._label.setText(t("Working…"))
            self.setVisible(True)
        elif n == 0 and self._depth == 0:
            self.setVisible(False)

    @contextmanager
    def working(self, message: str = ""):
        self.begin(message)
        try:
            yield
        finally:
            self.end()
