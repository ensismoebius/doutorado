"""MainWindow integration tests (ESPECIFICACAO_DLVL.md #24 Interface, #25 Controles).

Everything the render tests below the widget layer never exercise: view
routing when a demo is selected, frame/explanation label updates on
stepping, the fixed explanation-label height (regression test for the
prior session's fix), the deep-linking slug selector, and the keyboard
shortcuts -- the wiring that lives entirely in app/main_window.py.
"""

import pytest
from unittest.mock import Mock

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from efficient_nn_lab.app.main_window import MainWindow, _build_demo_tree, _choose_view


def _all_demos():
    demos = []
    for group in _build_demo_tree().values():
        demos.extend(group)
    return demos


def _window(qapp):
    """A shown, *activated* MainWindow ready to receive key events.

    The keyboard tests below depend on activation, not just visibility:
    MainWindow's shortcuts are plain QShortcuts, whose default context is
    ``Qt.WindowShortcut`` -- Qt only matches those when the shortcut's
    window is the *active* window. ``show()`` alone does not guarantee
    that; activation arrives as a later platform event, so whether it had
    landed by the time a test pressed a key came down to timing (these
    tests passed or failed run to run, and every earlier test in the
    session leaves its own window around to compete for activation).

    ``qWaitForWindowActive`` is the API for exactly this: it spins the
    event loop until the window really is active. It is asserted rather
    than best-effort, so a genuine activation failure surfaces here
    instead of resurfacing as a baffling "the shortcut did nothing"
    assertion further down the test.
    """
    window = MainWindow()
    window.show()
    window.activateWindow()
    assert QTest.qWaitForWindowActive(window), "window never became active"
    QTest.qWait(30)  # let the layout settle for the geometry assertions
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
    ["comparison", "bitnet.ste", "snn.poisson_image", "snn.lif"],
    ids=["comparison", "bitnet.ste", "snn.poisson_image", "snn.lif"],
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


def test_lif_canvas_not_shrunk_by_short_explanations(qapp):
    """Regression for "the SNN->LIF graphs are too small".

    LIF's explanations are short, but a global reservation sized for the
    wordiest demo made it pay for them anyway; together with its 4
    parameter sliders that shrank the canvas to ~280px. The reservation is
    per demo, so LIF only pays for its own text.

    Compared against a wordy demo measured in the same window rather than
    against a pixel count: the point is that the two differ, and hardcoded
    pixels would just re-break whenever the fonts or the text change.
    """
    from efficient_nn_lab.app.main_window import _MIN_CANVAS_HEIGHT

    window = _window(qapp)
    wordy = next(d for d in _all_demos() if d.slug == "snn.surrogate")
    window._select_demo(wordy)
    wordy_height = window.explanation_label.height()

    demo = next(d for d in _all_demos() if d.slug == "snn.lif")
    window._select_demo(demo)
    # a real window, not the 900x600 floor: at the minimum size the column
    # simply has less than 330px to give (see
    # test_animation_canvas_gets_everything_the_column_can_spare).
    window.resize(1400, 900)
    QTest.qWait(60)

    assert window.explanation_label.height() < wordy_height
    assert window.signal_view._canvas.height() >= _MIN_CANVAS_HEIGHT


def test_step_title_is_tall_enough_for_its_own_font(qapp):
    """The step title is 22pt bold; it used to be measured at 18pt.

    Sizing the frame-title label with the *explanation* label's metrics
    reserved 28px for a 34px line, so every step title on every demo lost
    its descenders. Silent failure: nothing errors, the text is just
    visibly cut off. This asserts each label clears one line of its OWN
    font -- the property that was violated.
    """
    window = _window(qapp)
    for demo in _all_demos():
        window._select_demo(demo)
        for label in (window.frame_label, window.explanation_label):
            spacing = QFontMetrics(label.font()).lineSpacing()
            assert label.height() >= spacing, (
                f"{demo.slug}: {label.objectName() or 'label'} reserves "
                f"{label.height()}px for a {spacing}px line"
            )


def test_step_title_never_clips_the_text_it_shows(qapp):
    """Every step title of every demo must fit the reserved height."""
    window = _window(qapp)
    for demo in _all_demos():
        window._select_demo(demo)
        reserved = window.frame_label.height()
        fm = QFontMetrics(window.frame_label.font())
        width = max(400, window.frame_label.width() - 6)
        for frame in demo.checkpoint_frames():
            text = window._frame_title_text(frame.label, 1, demo.total_steps)
            needed = fm.boundingRect(
                QRect(0, 0, width, 0), Qt.TextFlag.TextWordWrap, text
            ).height()
            assert needed <= reserved, (
                f"{demo.slug}: {text!r} needs {needed}px, only {reserved}px reserved"
            )


def test_deep_linked_demo_reserves_the_right_heights(qapp):
    """The --demo path measures before Qt has applied the stylesheet.

    MainWindow(slug) selects the demo from __init__, so the labels are
    measured while their fonts are still the 9pt default rather than the
    stylesheet's 18/22pt -- which reserved 24px for a 34px line on every
    demo opened from a slide link. ensurePolished() is what makes the
    measurement see the real font.
    """
    from efficient_nn_lab.app.main_window import MainWindow as _MainWindow

    window = _MainWindow("snn.lif")
    window.show()
    assert QTest.qWaitForWindowActive(window)
    QTest.qWait(30)
    for label in (window.frame_label, window.explanation_label):
        spacing = QFontMetrics(label.font()).lineSpacing()
        assert label.height() >= spacing, (
            f"deep link reserved {label.height()}px for a {spacing}px line"
        )


def test_reservations_are_remeasured_on_resize(qapp):
    """The reservation belongs to the window it was measured in.

    Both inputs change with the window: how many lines the text wraps to
    (width) and how many pixels the caption may claim (height, via
    _MAX_TEXT_SHARE). A reservation computed for a big window and left
    alone would, in a small one, exceed the budget and eat the canvas --
    which is exactly what the resize hook exists to prevent.
    """
    from efficient_nn_lab.app.main_window import _MAX_TEXT_SHARE

    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == "backprop.chain")
    window._select_demo(demo)

    for width, height in ((1180, 720), (920, 520), (1400, 900)):
        window.resize(width, height)
        QTest.qWait(60)
        reserved = window.frame_label.height() + window.explanation_label.height()
        budget = window.height() * _MAX_TEXT_SHARE
        assert reserved <= budget + 1, (
            f"at {window.width()}x{window.height()} the captions reserve {reserved}px, "
            f"over the {budget:.0f}px budget"
        )


def test_right_column_widgets_never_overlap(qapp):
    """Qt lays widgets ON TOP of each other when their minimums do not fit.

    This is how a well-meant floor breaks a window: ask for more than the
    column can give and nothing errors -- the step title simply paints over
    the animation. Nothing in the app notices, and no other test looks.
    """
    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == "backprop.chain")
    window._select_demo(demo)

    for width, height in ((900, 600), (900, 710), (1180, 720), (1400, 900)):
        window.resize(width, height)
        QTest.qWait(60)
        column = [
            window.demo_title_label, window.demo_description_label, window.stack,
            window.frame_label, window.explanation_label, window.controls,
        ]
        for upper, lower in zip(column, column[1:]):
            if not lower.isVisible():
                continue
            assert upper.y() + upper.height() <= lower.y(), (
                f"at {window.width()}x{window.height()}, {upper.objectName() or type(upper).__name__} "
                f"(ends {upper.y() + upper.height()}) overlaps "
                f"{lower.objectName() or type(lower).__name__} (starts {lower.y()})"
            )


def test_animation_canvas_gets_everything_the_column_can_spare(qapp):
    """Regression for "the graphs are too small", second round.

    Making the caption reservations *correct* made them bigger, and they
    took the pixels straight out of the animation: 59px of canvas at the
    minimum window size. The floor claims what is left after the chrome,
    so the canvas grows with the window until it reaches _MIN_CANVAS_HEIGHT
    and never drops under _ABS_MIN_CANVAS_HEIGHT.
    """
    from efficient_nn_lab.app.main_window import _ABS_MIN_CANVAS_HEIGHT, _MIN_CANVAS_HEIGHT

    window = _window(qapp)
    demo = next(d for d in _all_demos() if d.slug == "snn.lif")
    window._select_demo(demo)

    window.resize(900, 600)
    QTest.qWait(60)
    smallest = window.stack.height()
    assert smallest >= _ABS_MIN_CANVAS_HEIGHT, f"canvas is {smallest}px at the minimum window"

    window.resize(1400, 900)
    QTest.qWait(60)
    roomy = window.stack.height()
    assert roomy >= _MIN_CANVAS_HEIGHT, f"canvas is only {roomy}px in a 1400x900 window"
    assert roomy > smallest, "the canvas must take the room a bigger window gives it"


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


# -- "Loop rápido": continuous, dwell-free playback --------------------

def _loop_demo(window):
    demo = next(d for d in _all_demos() if d.slug == "snn.poisson_image")
    window._select_demo(demo)
    return demo


def test_fast_loop_button_shown_only_for_demos_that_offer_it(qapp):
    window = _window(qapp)
    _loop_demo(window)
    assert window.controls._loop_btn.isVisible()

    plain = next(d for d in _all_demos() if d.slug == "snn.lif")
    window._select_demo(plain)
    assert not window.controls._loop_btn.isVisible()
    assert not window.controls._loop_btn.isChecked()


def test_fast_loop_wraps_around_instead_of_stopping(qapp):
    """The behaviour that makes it a *loop*, driven deterministically.

    The tick is called directly instead of waiting on the QTimer: the point
    under test is the wrap decision, and a timing-based test of it would
    only add flakiness (see _window's docstring for what that costs).
    """
    window = _window(qapp)
    demo = _loop_demo(window)
    player = window.player
    n = len(demo._frames)

    player.play_fast_loop()
    assert player.is_looping
    assert demo.is_playing

    # Drive the tick by hand with the timer stopped: leaving it running
    # would race these calls (a redraw re-enters the event loop, so the
    # timer can fire between them) and the test would be measuring
    # scheduling, not the wrap decision it is about.
    player._timer.stop()
    seen = set()
    for _ in range(2 * n):
        seen.add(demo.current_frame_index)
        player._tick()

    assert seen == set(range(n)), "o loop não passou por todos os quadros"
    # crossed the end at least twice and never stopped
    assert player.is_looping
    assert demo.is_playing


def test_normal_play_still_stops_at_the_end(qapp):
    # the loop mode must not have turned ordinary Play into a loop.
    window = _window(qapp)
    demo = _loop_demo(window)
    player = window.player
    player.play()
    assert not player.is_looping
    player._timer.stop()
    for _ in range(len(demo._frames) + 5):
        player._tick()
    assert demo.is_at_last_frame()
    assert not demo.is_playing


def test_fast_loop_button_reflects_player_and_second_click_stops(qapp):
    window = _window(qapp)
    _loop_demo(window)
    btn = window.controls._loop_btn

    btn.click()
    assert window.player.is_looping
    assert btn.isChecked()
    assert window.controls._play_btn.text() == "Pause"

    btn.click()
    assert not window.player.is_looping
    assert not window.player.demo.is_playing
    assert not btn.isChecked()


def test_pause_and_reset_also_stop_the_fast_loop(qapp):
    for stop in ("_on_pause", "_on_reset"):
        window = _window(qapp)
        _loop_demo(window)
        window.controls._loop_btn.click()
        assert window.player.is_looping
        getattr(window, stop)()
        window._refresh_frame()
        assert not window.player.is_looping, stop
        assert not window.controls._loop_btn.isChecked(), stop


def test_refresh_frame_does_not_feed_the_loop_button_back_on_itself(qapp):
    # set_fast_loop_active runs on every frame refresh; if setChecked
    # re-emitted, the refresh would re-trigger the handler that caused it.
    window = _window(qapp)
    _loop_demo(window)
    window.controls._loop_btn.click()
    calls = []
    window.controls.fast_loop_clicked.connect(lambda: calls.append(1))
    for _ in range(5):
        window._refresh_frame()
    assert calls == []
    assert window.player.is_looping


def test_tick_is_not_re_entrant(qapp):
    """A redraw re-enters the Qt event loop, so the timer can fire mid-tick.

    Without the guard the demo advances twice per interval under a slow
    renderer -- playback silently runs at double speed, and nests deeper as
    rendering gets slower. Measured on snn.poisson_image: 8 hand-driven
    ticks executed 16 times before the guard.
    """
    window = _window(qapp)
    demo = _loop_demo(window)
    player = window.player
    player.play_fast_loop()

    executed = []
    inner = player._tick_once

    def counting():
        executed.append(1)
        inner()

    player._tick_once = counting
    indices = []
    for _ in range(8):
        indices.append(demo.current_frame_index)
        player._tick()

    assert len(executed) == 8, f"tick re-entrou: {len(executed)} execuções para 8 chamadas"
    assert indices == list(range(8)), indices


# -- lecture-mode navigation (FIXME.md) --------------------------------
#
# Lecture mode hides the demo tree, and the tree was the only way to move
# from one demo to the next -- so the mode built for presenting was the one
# mode you could not present in without leaving it. These tests pin the
# button that fixes that, plus the section-crossing order it has to follow.

def _slug_order(window):
    from efficient_nn_lab.app.main_window import _demo_order
    return [d.slug for d in _demo_order(window._demo_groups)]


def test_demo_order_is_exactly_the_tree_order(qapp):
    """The running order must be the sidebar's order, not a second list."""
    window = _window(qapp)
    from_tree = []
    for i in range(window.tree.topLevelItemCount()):
        parent = window.tree.topLevelItem(i)
        for j in range(parent.childCount()):
            payload = parent.child(j).data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(payload, str):
                from_tree.append(payload.slug)
    assert _slug_order(window) == from_tree


def test_next_demo_advances_inside_the_current_section(qapp):
    window = _window(qapp)
    order = _slug_order(window)
    window._select_demo_by_slug(order[0])
    window.next_demo_btn.click()
    assert window.player.demo.slug == order[1]


def test_next_demo_crosses_into_the_first_item_of_the_next_section(qapp):
    """The explicit requirement: last of a section -> first of the next."""
    window = _window(qapp)
    groups = list(window._demo_groups.values())
    last_of_first_section = groups[0][-1]
    first_of_next_section = groups[1][0]
    window._select_demo_by_slug(last_of_first_section.slug)
    window.next_demo_btn.click()
    assert window.player.demo.slug == first_of_next_section.slug


def test_next_demo_wraps_around_at_the_very_end(qapp):
    """A dead button at the end of the deck would only be found mid-talk."""
    window = _window(qapp)
    order = _slug_order(window)
    window._select_demo_by_slug(order[-1])
    window.next_demo_btn.click()
    assert window.player.demo.slug == order[0]


def test_next_demo_walks_every_demo_exactly_once_per_lap(qapp):
    window = _window(qapp)
    order = _slug_order(window)
    window._select_demo_by_slug(order[0])
    visited = [window.player.demo.slug]
    for _ in range(len(order) - 1):
        window.next_demo_btn.click()
        visited.append(window.player.demo.slug)
    assert visited == order


def test_next_demo_button_survives_lecture_mode(qapp):
    """The whole point: it must still be there once the tree is gone."""
    window = _window(qapp)
    window._select_demo_by_slug(_slug_order(window)[0])
    window.lecture_mode_btn.setChecked(True)
    QTest.qWait(20)
    assert not window.tree.isVisible(), "lecture mode should hide the tree"
    assert window.next_demo_btn.isVisible()
    assert window.next_demo_btn.isEnabled()
    window.next_demo_btn.click()
    assert window.player.demo.slug == _slug_order(window)[1]


def test_next_demo_moves_the_tree_highlight_too(qapp):
    """Leaving lecture mode must not reveal a sidebar pointing elsewhere."""
    window = _window(qapp)
    order = _slug_order(window)
    window._select_demo_by_slug(order[0])
    window.next_demo_btn.click()
    current = window.tree.currentItem()
    assert current is not None
    assert current.data(0, Qt.ItemDataRole.UserRole).slug == order[1]


def test_next_demo_from_the_welcome_screen_opens_the_first_demo(qapp):
    window = _window(qapp)
    assert window.player is None
    window._on_next_demo()
    assert window.player is not None
    assert window.player.demo.slug == _slug_order(window)[0]


def test_next_demo_button_is_disabled_until_something_is_selected(qapp):
    window = _window(qapp)
    assert not window.next_demo_btn.isEnabled()
    window._select_demo_by_slug(_slug_order(window)[0])
    assert window.next_demo_btn.isEnabled()


def test_n_key_advances_to_the_next_demo(qapp):
    """N, not Right/PageDown: Right steps *inside* a demo and presenter
    remotes send PageDown for the slides."""
    window = _window(qapp)
    order = _slug_order(window)
    window._select_demo_by_slug(order[0])
    QTest.keyClick(window, Qt.Key.Key_N)
    assert window.player.demo.slug == order[1]


def test_right_arrow_still_steps_inside_the_demo(qapp):
    """Guard against the shortcut for "next demo" stealing "next step"."""
    window = _window(qapp)
    order = _slug_order(window)
    window._select_demo_by_slug(order[0])
    before = window.player.demo.current_frame_index
    QTest.keyClick(window, Qt.Key.Key_Right)
    # StepPlayer animates the leg to the next checkpoint on its timer, so
    # the index moves over the following ticks rather than immediately.
    QTest.qWait(600)
    assert window.player.demo.slug == order[0]
    assert window.player.demo.current_frame_index > before
