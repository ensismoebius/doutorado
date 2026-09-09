"""Application-wide UI toggles that are not part of the pipeline selection.

Kept as a QObject (not a dataclass) for the same reason as
``efficient_nn_lab/core/state.py``: several panels react live to these flags.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AppState(QObject):
    """Cross-cutting UI state (FIXME §9, §26, §33, §38)."""

    follow_data_mode_changed = Signal(bool)
    low_performance_mode_changed = Signal(bool)
    display_downsampled_changed = Signal(bool)
    theme_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._follow_data_mode = True
        self._low_performance_mode = False
        self._display_downsampled = False
        self._theme = "system"

    @property
    def follow_data_mode(self) -> bool:
        """When on, selecting a sample exposes its whole processing chain
        (FIXME §9)."""
        return self._follow_data_mode

    @follow_data_mode.setter
    def follow_data_mode(self, value: bool) -> None:
        if value != self._follow_data_mode:
            self._follow_data_mode = value
            self.follow_data_mode_changed.emit(value)

    @property
    def low_performance_mode(self) -> bool:
        """When on, disable 3D, animation, anti-aliasing, live updates
        (FIXME §38)."""
        return self._low_performance_mode

    @low_performance_mode.setter
    def low_performance_mode(self, value: bool) -> None:
        if value != self._low_performance_mode:
            self._low_performance_mode = value
            self.low_performance_mode_changed.emit(value)

    @property
    def display_downsampled(self) -> bool:
        """True while any visible curve is decimated for rendering — the
        status bar shows DISPLAY-DOWNSAMPLED vs FULL RESOLUTION (FIXME §26)."""
        return self._display_downsampled

    @display_downsampled.setter
    def display_downsampled(self, value: bool) -> None:
        if value != self._display_downsampled:
            self._display_downsampled = value
            self.display_downsampled_changed.emit(value)

    @property
    def theme(self) -> str:
        return self._theme

    @theme.setter
    def theme(self, value: str) -> None:
        if value != self._theme:
            self._theme = value
            self.theme_changed.emit(value)
