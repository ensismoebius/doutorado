"""MainWindow integration tests (FIXME.md 5.3).

Everything the render tests below the widget layer never exercise: view
routing when a demo is selected, frame/explanation label updates on
stepping, the fixed explanation-label height (regression test for the
prior session's fix), the deep-linking slug selector, and the keyboard
shortcuts -- the wiring that lives entirely in app/main_window.py.
"""

import pytest
from unittest.mock import Mock

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from efficient_nn_lab.app.main_window import MainWindow, _build_demo_tree, _choose_view


def _all_demos():
    demos = []
    for group in _build_demo_tree().values():
        demos.extend(group)
    return demos


def _window(qapp):
    window = MainWindow()
    window.show()
    QTest.qWait(30)
    return window


# -- routing -----------------------------------------------------------

@pytest.mark.parametrize("demo", _all_demos(), ids=lambda d: d.slug)
def test_selecting_demo_routes_to_correct_stack_widget(qapp, demo):
    window = _window(qapp)
    expected = {
        "signal": window.signal_view,
        "weight": window.weight_view,
        "neuron": window.neuron_view,
    }[_choose_view(demo.current_frame().values)]
    window._select_demo(demo)
    assert window.stack.currentWidget() is expected
    assert window.demo_title_label.text() == demo.title
    assert window.controls.isEnabled()


def test_references_item_routes_to_references_view(qapp):
    window = _window(qapp)
    ref_item = window.tree.topLevelItem(window.tree.topLevelItemCount() - 1)
    assert ref_item is not None
    window._on_tree_item_clicked(ref_item, 0)
    assert window.stack.currentWidget() is window.references_view
    assert window.demo_title_label.text() == "Referências"
    assert not window.controls.isEnabled()


# -- label updates on stepping -----------------------------------------

@pytest.mark.parametrize(
    "slug, steps",
    [("backprop.mlp", 5), ("snn.surrogate", 5)],
)
def test_stepping_updates_frame_and_explanation_labels(qapp, slug, steps):
    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == slug)
    window._select_demo(demo)
    for _ in range(steps):
        demo.step_forward()
        window._refresh_frame()
        frame = demo.current_frame()
        assert window.frame_label.text().startswith(frame.label)
        assert window.explanation_label.text() == frame.explanation
        assert "passo" in window.frame_label.text().lower()


# -- fixed explanation-label height (regression) -----------------------

@pytest.mark.parametrize(
    "slug",
    ["comparison", "bitnet.ste", "snn.poisson_image"],
    ids=["comparison", "bitnet.ste", "snn.poisson_image"],
)
def test_explanation_label_height_constant_across_checkpoints(qapp, slug):
    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == slug)
    window._select_demo(demo)
    height = window.explanation_label.height()
    assert height > 0
    frame_height = window.frame_label.height()
    assert frame_height > 0
    for _ in range(demo.total_steps - 1):
        demo.step_forward()
        window._refresh_frame()
        assert window.explanation_label.height() == height
        assert window.frame_label.height() == frame_height


# -- keyboard shortcuts ------------------------------------------------

def test_keyboard_shortcuts_trigger_expected_slots(qapp):
    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == "bitnet.forward")
    window._select_demo(demo)
    assert window.player is not None
    player = window.player

    player.step_forward = Mock()
    player.step_backward = Mock()
    player.reset = Mock()
    QTest.keyClick(window, Qt.Key.Key_Right)
    player.step_forward.assert_called_once()
    QTest.keyClick(window, Qt.Key.Key_Left)
    player.step_backward.assert_called_once()
    QTest.keyClick(window, Qt.Key.Key_R)
    player.reset.assert_called_once()

    player.play = Mock()
    player.pause = Mock()
    demo.is_playing = False
    QTest.keyClick(window, Qt.Key.Key_Space)
    player.play.assert_called_once()
    demo.is_playing = True
    QTest.keyClick(window, Qt.Key.Key_Space)
    player.pause.assert_called_once()


def test_escape_returns_to_welcome(qapp):
    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == "comparison")
    window._select_demo(demo)
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert window.demo_title_label.text() == "Efficient Neural Networks Lab"
    assert not window.controls.isEnabled()
    assert window.stack.currentWidget() is not window.references_view


# -- deep linking ------------------------------------------------------

def test_select_demo_by_slug_valid(qapp):
    window = _window(qapp)
    window._select_demo_by_slug("comparison")
    assert window.player is not None
    assert window.player.demo.slug == "comparison"
    assert window.controls.isEnabled()


def test_select_demo_by_slug_unknown_is_ignored(qapp, capsys):
    window = _window(qapp)
    window._select_demo_by_slug("no.such.demo")
    assert window.player is None
    assert window.demo_title_label.text() == "Efficient Neural Networks Lab"
    err = capsys.readouterr().err
    assert "no.such.demo" in err
