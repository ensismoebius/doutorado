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

    The top panel is TWO pixel grids side by side (original | gutter |
    rate-decoded estimate), the bottom panel is ONE -- so "the whole pixel
    grid" is a *different* width for each axis; `full_top_xlim` and
    `full_bottom_xlim` are deliberately not the same value (unlike
    `full_ylim`, which both panels share: same `rows`).
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
    gutter = max(2, cols // 24)
    pair_cols = 2 * cols + gutter
    full_top_xlim = pytest.approx((-0.5, pair_cols - 0.5))
    full_bottom_xlim = pytest.approx((-0.5, cols - 0.5))
    full_ylim = pytest.approx((rows - 0.5, -0.5))
    first_top = (view._ax_top.get_xlim(), view._ax_top.get_ylim())

    # leave a different demo's (stale) limits behind on the same axes
    other_frame = next(
        f for f in SpikeGenerationDemo()._frames if f.values.get("kind") == "signal_spikes"
    )
    view.render(other_frame.values)
    assert tuple(view._ax_top.get_xlim()) != tuple(first_top[0])

    view.render(poisson_frame.values)
    assert view._ax_top.get_xlim() == full_top_xlim
    assert view._ax_top.get_ylim() == full_ylim
    assert view._ax_bottom.get_xlim() == full_bottom_xlim
    assert view._ax_bottom.get_ylim() == full_ylim
    assert tuple(view._ax_top.get_xlim()) == tuple(first_top[0])


def test_poisson_image_top_panel_is_not_narrowed_to_the_bottom_panels_range(qapp):
    """Regression for the actual "Patrick looks zoomed in" report: the two
    panels used to share an x-axis (`sharex=True` at construction), so
    whichever of the two `set_xlim` calls in `_render_poisson_image` ran
    *last* (the bottom panel's, narrower by exactly `cols + gutter`) won
    for *both* axes -- silently cropping the top panel's right half (the
    whole rate-decoded estimate) out of view, and the visible left half
    read as a tighter crop on Patrick's face than the source photo. The
    two panels must be free to hold genuinely different x-ranges.
    """
    from efficient_nn_lab.snn.demos.poisson_image_coding import PoissonImageCodingDemo

    view = SignalView()
    poisson = PoissonImageCodingDemo()
    poisson_frame = next(
        f for f in poisson._frames if f.values.get("kind") == "poisson_image_coding"
    )
    view.render(poisson_frame.values)

    image = np.asarray(poisson_frame.values["image"])
    rows, cols = image.shape
    gutter = max(2, cols // 24)
    pair_cols = 2 * cols + gutter
    top_width = view._ax_top.get_xlim()[1] - view._ax_top.get_xlim()[0]
    bottom_width = view._ax_bottom.get_xlim()[1] - view._ax_bottom.get_xlim()[0]
    assert top_width == pytest.approx(pair_cols)
    assert bottom_width == pytest.approx(cols)
    # the whole point: the estimate half must be inside the visible range,
    # not just a wider xlim number that still cuts it off some other way.
    assert view._ax_top.get_xlim()[1] >= cols + gutter + cols - 0.5 - 1e-6


def test_lif_x_axis_stays_fixed_to_the_full_window_as_the_trace_grows(qapp):
    """Regression: LIFDynamicsDemo's frames carry a growing *partial*
    trace (`current[:t+1]`), and `_render_lif` used to never call
    `set_xlim` at all -- relying on whatever the axes happened to have
    left over (stale limits from the previously-rendered demo, or plain
    autoscale-to-partial-data on a fresh widget). Either way the x-axis
    changed shape frame to frame instead of staying fixed to the demo's
    full window, which this file's own module docstring promises
    ("nothing about a leaking, integrating potential should ever jump").
    """
    from efficient_nn_lab.snn.demos.lif_dynamics import LIFDynamicsDemo

    view = SignalView()
    demo = LIFDynamicsDemo()
    n_total = len(demo._frames)

    view.render(demo._frames[5].values)  # an early, still-growing frame
    early_xlim = (view._ax_top.get_xlim(), view._ax_bottom.get_xlim())

    view.render(demo._frames[-1].values)  # the full trace
    late_xlim = (view._ax_top.get_xlim(), view._ax_bottom.get_xlim())

    assert early_xlim == late_xlim
    assert view._ax_top.get_xlim() == pytest.approx((0.0, n_total - 1))


def test_axis_title_and_aspect_do_not_leak_between_demo_kinds(qapp):
    """Regression: fast_clear() drops artists (lines, images, legends...)
    but a title and an aspect-ratio mode are Axes *properties*, not
    artists -- snn.poisson_image is the only kind that sets either
    (title on both panels; aspect="equal" so Patrick's pixels stay
    square), and every other kind's render() only ever sets what IT
    wants, on the unstated assumption that "not set" means "Matplotlib's
    default". On this widget's long-lived, reused axes that assumption
    used to be false: switching from snn.poisson_image (right before
    snn.lif in the sidebar's own running order) to any other demo left
    both its "Spikes sorteados..." title AND its aspect="equal" behind.
    The second one is the damaging one -- combined with
    adjustable="datalim" it makes Matplotlib silently override the next
    kind's own explicit set_ylim to keep pixels square, which is how a
    LIF membrane trace deliberately ranged to (-0.1, 1.4) rendered at
    roughly (-3.1, 4.4) instead: a wrong, arbitrary range that reads as
    "the curve barely moves".
    """
    from efficient_nn_lab.snn.demos.poisson_image_coding import PoissonImageCodingDemo
    from efficient_nn_lab.snn.demos.lif_dynamics import LIFDynamicsDemo

    view = SignalView()
    view.render(next(
        f for f in PoissonImageCodingDemo()._frames if f.values.get("kind") == "poisson_image_coding"
    ).values)
    assert view._ax_bottom.get_title() != ""  # sanity: poisson_image does set one

    lif_frame = next(f for f in LIFDynamicsDemo()._frames if f.values.get("kind") == "lif_trace")
    view.render(lif_frame.values)
    assert view._ax_top.get_title() == ""
    assert view._ax_bottom.get_title() == ""
    assert view._ax_top.get_aspect() == "auto"
    assert view._ax_bottom.get_aspect() == "auto"
    assert view._ax_bottom.get_ylim() == pytest.approx((-0.1, float(lif_frame.values["v_th"]) * 1.4))
