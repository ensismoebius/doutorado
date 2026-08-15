"""A faster substitute for ``Axes.clear()``.

Every widget here redraws its whole picture from scratch on every animation
frame (the "persistent scene" architecture -- see neuron_view.py's module
docstring), which used to mean calling ``ax.clear()`` every frame. That
call is deceptively expensive in matplotlib: it doesn't just drop the
plotted artists, it also tears down and rebuilds the X/Y ``Axis`` objects
(ticks, spines, transforms...), which profiling showed costing ~80ms per
call on this project's diagrams -- at StepPlayer's ~25fps target (40ms/
frame budget), a *single* ``clear()`` already blows the whole frame
budget, and some widgets call it more than once per frame. That is what
made the recently-added multi-panel demos (multilayer_network.py in
particular, with six axes redrawn per frame) visibly stutter.

``fast_clear`` removes only the artists actually drawn each frame (lines,
patches, text, collections, the legend) and leaves the Axis machinery
alone. Every render() method here already sets xlim/ylim/labels/title
explicitly on every call, so nothing depended on ``clear()``'s side effect
of resetting those -- this is a drop-in replacement.
"""

from __future__ import annotations


def fast_clear(ax) -> None:
    for artist in list(ax.lines):
        artist.remove()
    for artist in list(ax.patches):
        artist.remove()
    for artist in list(ax.texts):
        artist.remove()
    for artist in list(ax.collections):
        artist.remove()
    for artist in list(ax.artists):
        artist.remove()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
