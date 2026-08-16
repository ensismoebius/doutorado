"""No animation artist may ever render outside its widget's visible canvas.

FigureCanvasQTAgg rescales a figure's effective dpi to match whatever pixel
size Qt gives the widget, so ordinary window resizing does not literally
clip content (verified this session: canvas sizes from 909x401 down to a
deliberately extreme 615x59 all produced zero overflow) -- but a genuine
coordinate bug (a box/text placed past its axes' declared xlim/ylim by more
than the axes' own margin can absorb) still would. This test renders every
checkpoint frame of every demo at two canvas sizes -- a generous one and a
cramped one below MainWindow's setMinimumSize floor -- and asserts nothing
in the fully-rendered figure extends past the canvas bounds.
"""

from __future__ import annotations

import pytest

from efficient_nn_lab.app.main_window import _build_demo_tree, _choose_view
from efficient_nn_lab.widgets.neuron_view import NeuronView
from efficient_nn_lab.widgets.signal_view import SignalView
from efficient_nn_lab.widgets.weight_view import WeightView

#: A comfortable size (roughly what MainWindow gives the canvas at its
#: default 1180x720 window) and a stress size well below the app's
#: setMinimumSize(900, 600) floor -- if this app is ever embedded/resized
#: outside MainWindow's own floor, content still must not clip.
_SIZES = [(900, 400), (400, 150)]

#: Pixel tolerance for the "did anything render outside the canvas" check --
#: matplotlib's own hinting/antialiasing can round an artist a fraction of
#: a pixel past its nominal bounds without it being visually clipped.
_MARGIN_PX = 2.0


def _all_demos():
    demos = []
    for group in _build_demo_tree().values():
        demos.extend(group)
    return demos


@pytest.mark.parametrize("demo", _all_demos(), ids=lambda d: d.title)
def test_no_checkpoint_renders_outside_its_canvas(qapp, demo):
    views = {"signal": SignalView(), "weight": WeightView(), "neuron": NeuronView()}
    for view in views.values():
        view.show()

    for width, height in _SIZES:
        for view in views.values():
            view.resize(width, height)

        for f in demo.checkpoint_frames():
            view_name = _choose_view(f.values)
            view = views[view_name]
            view.render(f.values)
            view._canvas.draw()

            fig = view._figure
            canvas_w, canvas_h = fig.bbox.width, fig.bbox.height
            tight = fig.get_tightbbox(view._canvas.get_renderer())

            assert tight.x0 >= -_MARGIN_PX, (
                f"{demo.title!r} frame {f.label!r} at {width}x{height}: "
                f"artist extends {-_MARGIN_PX - tight.x0:.1f}px past the left edge"
            )
            assert tight.y0 >= -_MARGIN_PX, (
                f"{demo.title!r} frame {f.label!r} at {width}x{height}: "
                f"artist extends {-_MARGIN_PX - tight.y0:.1f}px past the bottom edge"
            )
            assert tight.x1 <= canvas_w + _MARGIN_PX, (
                f"{demo.title!r} frame {f.label!r} at {width}x{height}: "
                f"artist extends {tight.x1 - canvas_w - _MARGIN_PX:.1f}px past the right edge"
            )
            assert tight.y1 <= canvas_h + _MARGIN_PX, (
                f"{demo.title!r} frame {f.label!r} at {width}x{height}: "
                f"artist extends {tight.y1 - canvas_h - _MARGIN_PX:.1f}px past the top edge"
            )
