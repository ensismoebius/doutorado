"""Embedded PyVista/VTK panel (FIXME §11, §39).

3D is opt-in and degradable: if PyVista, VTK or a usable GL context is missing
the panel falls back to a plain label instead of taking the window down with
it. Under ``QT_QPA_PLATFORM=offscreen`` (tests, CI) PyVista is put in
``OFF_SCREEN`` mode so a plotter can still be constructed and driven headless.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

PV_OK = False
_PV_ERR = ""
try:  # pragma: no cover - import guard
    import pyvista as pv
    from pyvistaqt import QtInteractor

    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pv.OFF_SCREEN = True
    PV_OK = True
except Exception as exc:  # noqa: BLE001
    pv = None  # type: ignore[assignment]
    QtInteractor = None  # type: ignore[assignment]
    _PV_ERR = f"{type(exc).__name__}: {exc}"


def import_error() -> str:
    return _PV_ERR


class PyVistaPanel(QWidget):
    """A `QtInteractor` plus a no-3D fallback."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        if not PV_OK:
            layout.addWidget(
                QLabel(
                    "3D view unavailable — PyVista/VTK did not import:\n"
                    f"    {_PV_ERR}\n"
                    "Install pyvista + pyvistaqt + vtk, or use the 2D panels."
                )
            )
            return
        try:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter.interactor)
        except Exception as exc:  # noqa: BLE001 - GL context missing etc.
            self.plotter = None
            layout.addWidget(QLabel(f"3D view could not start a render context:\n    {exc}"))

    @property
    def available(self) -> bool:
        return self.plotter is not None

    def clear(self) -> None:
        if self.plotter is not None:
            self.plotter.clear()

    def close(self) -> None:  # noqa: D401 - Qt override-ish
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:  # noqa: BLE001
                pass
        super().close()
