"""Every frame of every demo must render without raising, on whichever
widget main_window routes it to. Runs over *every* raw frame (checkpoints
and tweens alike), since the tween frames — carrying interpolated
values — are exactly where a widget assuming a field is never partially
blended, or never None, would break.
"""

import pytest
import numpy as np

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


def test_poisson_image_always_shows_full_photo_after_switching_demos(qapp):
    """Regression: the long-lived SignalView kept the xlim/ylim left by the
    previous demo, so returning to the Patrick photo showed only a zoomed-in
    corner. The view must be pinned to the whole pixel grid on every render.
    """
    from efficient_nn_lab.snn.demos.poisson_image_coding import PoissonImageCodingDemo
    from efficient_nn_lab.snn.demos.spike_generation import SpikeGenerationDemo

    view = SignalView()
    poisson = PoissonImageCodingDemo()
    poisson_frame = next(
        f for f in poisson._frames if f.values.get("kind") == "poisson_image_coding"
    )

    view.render(poisson_frame.values)
    image = np.asarray(poisson_frame.values["image"])
    rows, cols = image.shape
    full_xlim = pytest.approx((-0.5, cols - 0.5))
    full_ylim = pytest.approx((rows - 0.5, -0.5))
    first_top = (view._ax_top.get_xlim(), view._ax_top.get_ylim())

    # leave a different demo's (stale) limits behind on the same axes
    other_frame = next(
        f for f in SpikeGenerationDemo()._frames if f.values.get("kind") == "signal_spikes"
    )
    view.render(other_frame.values)
    assert tuple(view._ax_top.get_xlim()) != tuple(first_top[0])

    view.render(poisson_frame.values)
    assert view._ax_top.get_xlim() == full_xlim
    assert view._ax_top.get_ylim() == full_ylim
    assert view._ax_bottom.get_xlim() == full_xlim
    assert view._ax_bottom.get_ylim() == full_ylim
    assert tuple(view._ax_top.get_xlim()) == tuple(first_top[0])
    assert tuple(view._ax_top.get_ylim()) == tuple(first_top[1])
