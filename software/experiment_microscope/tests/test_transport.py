"""Shared animation transport (FIXME §24): player <-> bar <-> triangle sample index."""

from __future__ import annotations

from experiment_microscope.core.animation import TimelinePlayer
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.transport_bar import TransportBar
from experiment_microscope.views.triangle_view import TriangleView


def test_bar_follows_player(qapp):
    p = TimelinePlayer()
    bar = TransportBar(p)
    p.set_total_frames(10)
    assert bar._slider.maximum() == 9
    p.seek(4)
    assert bar._slider.value() == 4
    assert bar._pos.text() == "4 / 9"


def test_bar_disabled_when_single_frame(qapp):
    p = TimelinePlayer()
    bar = TransportBar(p)
    assert not bar.isEnabled()
    p.set_total_frames(5)
    assert bar.isEnabled()


def test_triangle_frame_scrubs_sample_index(qapp):
    p = TimelinePlayer()
    tri = TriangleView(DataRepository())
    tri.set_timeline(p)
    # no run node bound -> frame changes are ignored, no crash
    p.set_total_frames(20)
    p.seek(7)
    assert tri._sample.value() == 0

    # simulate a bound run
    tri._run_node = object()
    tri._sample.blockSignals(True)
    tri._sample.setRange(0, 19)
    tri._sample.blockSignals(False)
    p.seek(11)
    assert tri._sample.value() == 11


def test_triangle_spin_seeks_player_without_feedback_loop(qapp):
    p = TimelinePlayer()
    tri = TriangleView(DataRepository())
    tri.set_timeline(p)
    p.set_total_frames(20)
    tri._run_node = object()
    tri._sample.setRange(0, 19)
    tri._sample.setValue(6)
    assert p.current_frame == 6
