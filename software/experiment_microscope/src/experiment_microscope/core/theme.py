"""Dark / light / system theming (FIXME §34).

One entry point, ``apply_theme(app, name)``. It sets:

* the Qt application palette + a small stylesheet (takes effect immediately on
  every existing widget),
* pyqtgraph's global background / foreground (new plots; existing plots are
  restyled on their next ``show_node`` render),
* PyVista's plot theme (new 3D scenes).

``"system"`` follows the OS palette (dark if the default window colour is dark).
Colour is never the *only* signal in any view — line style, markers and labels
carry the same information (FIXME §34) — so switching themes only changes
comfort, not legibility.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette

_DARK = {
    "window": "#232629", "base": "#1b1e20", "text": "#e6e6e6",
    "button": "#2d3135", "highlight": "#3d7eff", "mid": "#3a3f44",
}
_LIGHT = {
    "window": "#f4f4f4", "base": "#ffffff", "text": "#1a1a1a",
    "button": "#e8e8e8", "highlight": "#2f6fed", "mid": "#c8c8c8",
}


def _is_system_dark(app) -> bool:
    c = app.palette().color(QPalette.ColorRole.Window)
    return c.lightness() < 128


def effective_theme(app, name: str) -> str:
    if name == "system":
        return "dark" if _is_system_dark(app) else "light"
    return "dark" if name == "dark" else "light"


def apply_theme(app, name: str) -> str:
    """Apply ``name`` (``system`` | ``light`` | ``dark``); returns the effective one."""
    eff = effective_theme(app, name)
    pal = _DARK if eff == "dark" else _LIGHT
    qpal = QPalette()
    qpal.setColor(QPalette.ColorRole.Window, QColor(pal["window"]))
    qpal.setColor(QPalette.ColorRole.Base, QColor(pal["base"]))
    qpal.setColor(QPalette.ColorRole.AlternateBase, QColor(pal["mid"]))
    qpal.setColor(QPalette.ColorRole.Text, QColor(pal["text"]))
    qpal.setColor(QPalette.ColorRole.WindowText, QColor(pal["text"]))
    qpal.setColor(QPalette.ColorRole.ButtonText, QColor(pal["text"]))
    qpal.setColor(QPalette.ColorRole.Button, QColor(pal["button"]))
    qpal.setColor(QPalette.ColorRole.ToolTipBase, QColor(pal["base"]))
    qpal.setColor(QPalette.ColorRole.ToolTipText, QColor(pal["text"]))
    qpal.setColor(QPalette.ColorRole.Highlight, QColor(pal["highlight"]))
    qpal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(qpal)
    app.setStyleSheet(
        f"QToolTip {{ color: {pal['text']}; background: {pal['base']}; "
        f"border: 1px solid {pal['mid']}; }}"
    )

    try:  # pyqtgraph is optional
        import pyqtgraph as pg

        pg.setConfigOption("background", pal["base"])
        pg.setConfigOption("foreground", pal["text"])
    except Exception:  # noqa: BLE001
        pass
    try:  # pyvista is optional
        import pyvista as pv

        pv.set_plot_theme("dark" if eff == "dark" else "document")
    except Exception:  # noqa: BLE001
        pass
    return eff
