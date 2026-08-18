"""The ANN x BitNet x SNN table must be readable from the back of a room.

FIXME.md: "the SNN and ANN comparison fonts are too small". The cause was
a hard-coded fontsize on a widget whose canvas grows: the Qt matplotlib
backend keeps dpi fixed and grows the figure's inches, so fixed point
sizes shrink *relatively* as the window grows -- worst exactly when the
app is projected. neuron_view.py now derives the type size from the
canvas and the column width instead.

These tests pin the three properties that fix has to keep, because none of
them is visible from reading the source:

  1. the type actually grows with the canvas (not "is big at one size"),
  2. nothing overlaps at any of those sizes -- bigger type is only an
     improvement if the cells still do not collide,
  3. the table fills the box it is given, instead of bunching against the
     top edge and leaving a third of the canvas blank.

Failure mode without them: silent. The demo renders, the tests that only
check "did it raise" pass, and the audience cannot read the table.
"""

from __future__ import annotations

import textwrap

import pytest

from efficient_nn_lab.comparison.ann_bitnet_snn import AnnBitnetSnnComparisonDemo
from efficient_nn_lab.widgets.neuron_view import NeuronView

#: The size the app actually hands this widget at a normal window size
#: (measured: MainWindow -> stack -> canvas), plus a smaller one.
_APP_SIZE = (1425, 1012)
_SMALL_SIZE = (900, 400)
_SIZES = [_APP_SIZE, (1200, 700), _SMALL_SIZE]

#: Cell type must be at least this fraction of the canvas height. At the
#: app's real size the old fontsize=8 was 0.8% -- unreadable on a
#: projector; the derived size is ~2.2%.
_MIN_TYPE_FRACTION = 0.018

#: The rendered table must span at least this fraction of the canvas
#: height. The first "bigger fonts" attempt scored 0.72 and left the
#: bottom third empty.
_MIN_FILL_FRACTION = 0.85


def _last_frame():
    """The final checkpoint: every row, the outputs row and both notes."""
    return AnnBitnetSnnComparisonDemo().checkpoint_frames()[-1]


def _rendered_texts(qapp, size):
    """(bbox_in_pixels, text, fontsize) for every visible text artist."""
    view = NeuronView()
    view.show()
    view.resize(*size)
    view.render(_last_frame().values)
    view._canvas.draw()
    renderer = view._canvas.get_renderer()
    out = []
    for artist in view._ax.texts:
        alpha = artist.get_alpha()
        if alpha is not None and alpha <= 0.02:
            continue
        out.append((artist.get_window_extent(renderer=renderer), artist.get_text(), artist.get_fontsize()))
    assert out, "the comparison table rendered no text at all"
    return view, out


@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_no_two_table_entries_overlap(qapp, size):
    _view, items = _rendered_texts(qapp, size)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, text_a, _ = items[i]
            b, text_b, _ = items[j]
            # 1px of tolerance for antialiasing/hinting; anything more is
            # two strings genuinely sitting on top of each other.
            overlap_w = min(a.x1, b.x1) - max(a.x0, b.x0)
            overlap_h = min(a.y1, b.y1) - max(a.y0, b.y0)
            assert not (overlap_w > 1.0 and overlap_h > 1.0), (
                f"at {size[0]}x{size[1]}, {text_a!r} and {text_b!r} overlap by "
                f"{overlap_w:.1f}x{overlap_h:.1f}px"
            )


def test_type_size_grows_with_the_canvas(qapp):
    """The bug this file exists for: fixed point size on a growing canvas."""
    _small_view, small = _rendered_texts(qapp, _SMALL_SIZE)
    _big_view, big = _rendered_texts(qapp, _APP_SIZE)
    small_fs = max(fs for _, _, fs in small)
    big_fs = max(fs for _, _, fs in big)
    assert big_fs > small_fs * 1.5, (
        f"type barely grew with the canvas: {small_fs:.1f}pt at {_SMALL_SIZE} vs "
        f"{big_fs:.1f}pt at {_APP_SIZE} -- a fixed fontsize would score 1.0x"
    )


@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_cell_type_is_a_readable_fraction_of_the_canvas(qapp, size):
    view, items = _rendered_texts(qapp, size)
    canvas_h_pt = view._figure.get_size_inches()[1] * 72.0
    # the row labels and cells share one size; the smallest of them is the
    # one that has to stay readable.
    body = [fs for bbox, text, fs in items if "Tipo de gradiente" not in text and "Eficiência não" not in text]
    smallest = min(body)
    fraction = smallest / canvas_h_pt
    assert fraction >= _MIN_TYPE_FRACTION, (
        f"at {size[0]}x{size[1]} the smallest cell is {smallest:.1f}pt = "
        f"{fraction * 100:.1f}% of the canvas height (floor {_MIN_TYPE_FRACTION * 100:.1f}%)"
    )


@pytest.mark.parametrize("size", _SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_the_table_fills_the_canvas_it_is_given(qapp, size):
    view, items = _rendered_texts(qapp, size)
    top = max(bbox.y1 for bbox, _, _ in items)
    bottom = min(bbox.y0 for bbox, _, _ in items)
    canvas_h = view._figure.bbox.height
    fill = (top - bottom) / canvas_h
    assert fill >= _MIN_FILL_FRACTION, (
        f"at {size[0]}x{size[1]} the table spans only {fill * 100:.0f}% of the canvas "
        f"height (floor {_MIN_FILL_FRACTION * 100:.0f}%) -- it is bunched at one edge"
    )


def test_every_cell_wraps_to_at_most_two_lines():
    """The row-height budget assumes two lines; three would collide.

    Pure text check, no rendering: it guards the *reason* _CMP_WRAP is 19
    rather than something smaller, so shortening it (or adding a longer
    cell to the demo) fails here instead of silently overlapping rows.
    """
    demo = AnnBitnetSnnComparisonDemo()
    table = demo.checkpoint_frames()[-1].values["table"]
    for row_name, cells in table.items():
        for cell in cells:
            lines = textwrap.fill(cell, NeuronView._CMP_WRAP).count("\n") + 1
            assert lines <= 2, f"{row_name!r} cell {cell!r} wraps to {lines} lines at width {NeuronView._CMP_WRAP}"
        label_lines = textwrap.fill(row_name, NeuronView._CMP_LABEL_WRAP).count("\n") + 1
        assert label_lines <= 2, f"row label {row_name!r} wraps to {label_lines} lines"


def test_the_outputs_row_is_labelled_with_a_word(qapp):
    """It used to say "O", which abbreviates nothing a student can decode."""
    _view, items = _rendered_texts(qapp, _APP_SIZE)
    labels = {text for _, text, _ in items}
    assert "Saída" in labels, f"no 'Saída' label among {sorted(labels)}"
