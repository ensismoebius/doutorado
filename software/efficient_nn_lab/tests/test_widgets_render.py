"""Every frame of every demo must render without raising, on whichever
widget main_window routes it to. Runs over *every* raw frame (checkpoints
and tweens alike), since the tween frames — carrying interpolated
values — are exactly where a widget assuming a field is never partially
blended, or never None, would break.
"""

import pytest

from efficient_nn_lab.app.main_window import _build_demo_tree, _choose_view
from efficient_nn_lab.widgets.neuron_view import NeuronView
from efficient_nn_lab.widgets.signal_view import SignalView
from efficient_nn_lab.widgets.weight_view import WeightView


def _all_demos():
    demos = []
    for group in _build_demo_tree().values():
        demos.extend(group)
    return demos


@pytest.mark.parametrize("demo", _all_demos(), ids=lambda d: d.title)
def test_every_raw_frame_renders_without_raising(qapp, demo):
    views = {"signal": SignalView(), "weight": WeightView(), "neuron": NeuronView()}
    for frame in demo._frames:
        view_name = _choose_view(frame.values)
        views[view_name].render(frame.values)


@pytest.mark.parametrize("demo", _all_demos(), ids=lambda d: d.title)
def test_stepping_through_checkpoints_renders_without_raising(qapp, demo):
    views = {"signal": SignalView(), "weight": WeightView(), "neuron": NeuronView()}
    for _ in range(demo.total_steps):
        frame = demo.current_frame()
        view_name = _choose_view(frame.values)
        views[view_name].render(frame.values)
        demo.step_forward()
