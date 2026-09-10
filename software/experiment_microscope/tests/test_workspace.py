"""Offscreen smoke test: the workspace opens and the explorer populates."""

from experiment_microscope.app.workspace import Workspace
from experiment_microscope.core.animation import TimelinePlayer
from experiment_microscope.core.i18n import t


def test_workspace_opens(qapp):
    w = Workspace()
    try:
        assert w.explorer.topLevelItemCount() >= 1  # one root per adapter
        w.para_plane.refresh()  # must not raise
        w.explorer.select_experiment("meeting01")
    finally:
        w.close()


def test_low_performance_mode_toggle(qapp):
    w = Workspace()
    try:
        assert not w.app_state.low_performance_mode
        w._low_perf_action.setChecked(True)
        assert w.app_state.low_performance_mode
        assert not w.transport.isEnabled()
        assert w._status_res.text() == t("LOW-PERFORMANCE MODE")
        w._low_perf_action.setChecked(False)
        assert not w.app_state.low_performance_mode
        assert w._status_res.text() == t("FULL RESOLUTION")
    finally:
        w.close()


def test_presentation_mode_hides_chrome_and_restores(qapp):
    w = Workspace()
    try:
        w._set_presentation_mode(True)
        assert w.app_state.presentation_mode
        assert not w.menuBar().isVisible()
        assert not w.follow_bar.isVisible()
        assert not any(d.isVisible() for d in w._hidden_docks)
        w._presenter_escape()
        assert not w.app_state.presentation_mode
        assert w.menuBar().isVisible()
        assert w.follow_bar.isVisible()
    finally:
        w._set_presentation_mode(False)
        w.close()


def test_presenter_shortcuts_drive_timeline(qapp):
    w = Workspace()
    try:
        w.timeline.set_total_frames(10)
        w._presenter_advance()
        w._presenter_advance()
        assert w.timeline.current_frame == 2
        w._presenter_back()
        assert w.timeline.current_frame == 1
    finally:
        w.close()


def test_cli_initial_tab_and_tour(qapp):
    w = Workspace(initial_tab="Wavelet Lab")
    try:
        assert w._tab_key(w.tabs.currentIndex()) == "Wavelet Lab"
    finally:
        w.close()
    w2 = Workspace(initial_tour="meeting01")
    try:
        assert w2._tour_open()
    finally:
        w2._end_tour()
        w2.close()


def test_right_docks_are_one_tab_group(qapp):
    w = Workspace()
    try:
        # every visible right dock is tabbed with the Inspector, so only one
        # occupies width at a time on a small screen
        prov = w._right_docks[0]
        group = set(w.tabifiedDockWidgets(prov))
        for d in w._right_docks[1:]:
            if not d.isHidden():
                assert d in group
    finally:
        w.close()


def test_reset_layout_restores_usable_docks(qapp):
    from PySide6.QtCore import Qt

    w = Workspace()
    try:
        w.resizeDocks([w._left_dock], [12], Qt.Orientation.Horizontal)
        w._right_docks[2].setFloating(True)
        w._reset_layout()
        assert not w._left_dock.isHidden() and not w._left_dock.isFloating()
        assert not w._right_docks[0].isHidden()
        assert not w._right_docks[2].isFloating()
    finally:
        w.close()


def test_timeline_player_basic(qapp):
    p = TimelinePlayer()
    frames = []
    p.frame_changed.connect(frames.append)
    p.set_total_frames(5)
    p.step_forward()
    p.step_forward()
    assert p.current_frame == 2
    p.step_backward()
    assert p.current_frame == 1
    p.seek(99)
    assert p.current_frame == 4
    assert frames  # emitted
