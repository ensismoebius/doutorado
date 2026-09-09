import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPalette

from experiment_microscope.core.theme import apply_theme, effective_theme


def test_apply_theme_switches_palette(qapp):
    apply_theme(qapp, "dark")
    dark = qapp.palette().color(QPalette.ColorRole.Window).lightness()
    apply_theme(qapp, "light")
    light = qapp.palette().color(QPalette.ColorRole.Window).lightness()
    assert dark < light
    assert effective_theme(qapp, "dark") == "dark"
    assert effective_theme(qapp, "light") == "light"
    assert effective_theme(qapp, "system") in ("dark", "light")


def test_apply_theme_returns_effective(qapp):
    assert apply_theme(qapp, "dark") == "dark"
    assert apply_theme(qapp, "system") in ("dark", "light")
