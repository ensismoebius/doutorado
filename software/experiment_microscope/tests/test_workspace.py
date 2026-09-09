"""Offscreen smoke test: the workspace opens and the explorer populates."""

from experiment_microscope.app.workspace import Workspace
from experiment_microscope.core.animation import TimelinePlayer


def test_workspace_opens(qapp):
    w = Workspace()
    try:
        assert w.explorer.topLevelItemCount() >= 1  # one root per adapter
        w.para_plane.refresh()  # must not raise
        w.explorer.select_experiment("meeting01")
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
