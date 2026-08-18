"""The chain-rule ladder has to stay readable and unambiguous.

Two complaints drove this demo's layout, and neither is visible from
reading the source, so both are pinned here:

  * "the fonts are too small, almost can't read" -- every box in this
    drawing is sized in DATA units and therefore scales with the widget,
    while a point size does not. The renderer scales its type by the
    canvas (`NeuronView._cl_type_scale`); these tests check the type is a
    readable fraction of the canvas *and* that it actually tracks it.
  * "the split lines are almost invisible, not clear what belongs to
    what" -- the local-derivative cards belong to the block above them,
    which is now said with a shaded column per block. A column is only
    an answer if the things inside it really are inside it, so the
    geometry test below checks each card sits in its block's column.

Failure mode without these: silent. Everything renders, the contract
tests pass, and the audience cannot read (or cannot parse) the screen.
"""

from __future__ import annotations

import itertools

import pytest

from efficient_nn_lab.backprop.demos.chain_rule_layers import ChainRuleLayersDemo
from efficient_nn_lab.widgets.neuron_view import NeuronView

#: The size the app actually gives this widget at a normal window size,
#: plus a smaller and a larger one.
_APP_SIZE = (1425, 1012)
_SIZES = [_APP_SIZE, (1200, 700), (1700, 950)]

#: The smallest revealed text must be at least this fraction of the canvas
#: height. The first version of this layout scored 0.009 (9pt of a 1012px
#: canvas) -- which is the "almost can't read" complaint, quantified.
_MIN_TYPE_FRACTION = 0.014


#: One view per canvas size, reused across frames: creating and showing a
#: NeuronView is far more expensive than rendering into an existing one,
#: and this module renders every checkpoint at every size.
_VIEWS: dict[tuple[int, int], NeuronView] = {}


def _rendered(size, frame):
    view = _VIEWS.get(size)
    if view is None:
        view = NeuronView()
        view.show()
        view.resize(*size)
        _VIEWS[size] = view
    view.render(frame.values)
    view._canvas.draw()
    renderer = view._canvas.get_renderer()
    items = []
    for artist in view._ax.texts:
        alpha = artist.get_alpha()
        if alpha is not None and alpha < 0.5:  # skeleton/faint notes
            continue
        items.append((artist.get_window_extent(renderer=renderer), artist.get_text(), artist.get_fontsize()))
    return view, items


def _checkpoints():
    return ChainRuleLayersDemo().checkpoint_frames()


@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_nothing_overlaps_at_any_checkpoint(qapp, size):
    for index, frame in enumerate(_checkpoints()):
        _view, items = _rendered(size, frame)
        for (box_a, text_a, _), (box_b, text_b, _) in itertools.combinations(items, 2):
            overlap_w = min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0)
            overlap_h = min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0)
            assert not (overlap_w > 2.0 and overlap_h > 2.0), (
                f"checkpoint {index} at {size[0]}x{size[1]}: {text_a!r} and {text_b!r} "
                f"overlap by {overlap_w:.0f}x{overlap_h:.0f}px"
            )


@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_type_is_a_readable_fraction_of_the_canvas(qapp, size):
    view, items = _rendered(size, _checkpoints()[-1])
    canvas_h_pt = view._figure.get_size_inches()[1] * 72.0
    smallest = min(fs for _, _, fs in items)
    fraction = smallest / canvas_h_pt
    assert fraction >= _MIN_TYPE_FRACTION, (
        f"at {size[0]}x{size[1]} the smallest text is {smallest:.1f}pt = "
        f"{fraction * 100:.2f}% of the canvas height (floor {_MIN_TYPE_FRACTION * 100:.1f}%)"
    )


def test_type_tracks_the_canvas_size(qapp):
    """A fixed point size would score 1.0x here; the boxes are not fixed."""
    _small_view, small = _rendered((900, 520), _checkpoints()[-1])
    _big_view, big = _rendered((1700, 950), _checkpoints()[-1])
    ratio = max(fs for _, _, fs in big) / max(fs for _, _, fs in small)
    assert ratio > 1.5, f"type barely grew with the canvas ({ratio:.2f}x)"


def test_every_card_sits_in_the_column_of_its_own_block(qapp):
    """The demo's claim, as geometry: one block, one column, its factors.

    The shaded column is what answers "which card belongs to which
    block?"; a card drifting outside its column's width would make the
    shading lie.
    """
    half = NeuronView._CL_COL_PITCH / 2
    for block, _row, *_ in NeuronView._CL_CARDS:
        centre = NeuronView._CL_NODE_X[block]
        left = centre - NeuronView._CL_CARD_W / 2
        right = centre + NeuronView._CL_CARD_W / 2
        assert left >= centre - half and right <= centre + half, (
            f"a card of block {block} is wider than its column"
        )


def test_the_drawing_stays_inside_its_canvas(qapp):
    """Catches a box (like the product chip) running off the right edge."""
    checkpoints = _checkpoints()
    # first / middle / last: what is drawn changes with the reveals, but
    # the skeleton (which is what defines the outer bounds) does not.
    sampled = [checkpoints[0], checkpoints[len(checkpoints) // 2], checkpoints[-1]]
    for index, frame in enumerate(sampled):
        view, _items = _rendered(_APP_SIZE, frame)
        figure = view._figure
        tight = figure.get_tightbbox(view._canvas.get_renderer())
        assert tight.x0 >= -2 and tight.y0 >= -2, f"checkpoint {index} spills past the left/bottom edge"
        assert tight.x1 <= figure.bbox.width + 2, f"checkpoint {index} spills past the right edge"
        assert tight.y1 <= figure.bbox.height + 2, f"checkpoint {index} spills past the top edge"
