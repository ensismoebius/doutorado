"""Publication-figure export (FIXME §30).

Interactive plotting stays in pyqtgraph; matplotlib is used only here, to
render a static figure for a paper. Views that hold exportable data implement
``export_figure(path, **opts)`` and build their figure through the helpers
below so every export shares the same defaults (vector-friendly, embedded
fonts, tight bounding box).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SUPPORTED = (".png", ".pdf", ".svg")


def new_figure(width_in: float = 6.0, height_in: float = 3.5, dpi: int = 300):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    fig.set_layout_engine("tight")
    return fig


def save_figure(fig, path: str | Path, *, transparent: bool = False) -> Path:
    path = Path(path)
    if path.suffix.lower() not in _SUPPORTED:
        raise ValueError(
            f"unsupported export format '{path.suffix}'. Use one of {_SUPPORTED}."
        )
    # Agg/PDF/SVG backends are import-time selected by the file extension via
    # Figure.savefig; no pyplot, no global state.
    fig.savefig(
        path,
        dpi=fig.get_dpi(),
        bbox_inches="tight",
        transparent=transparent,
        metadata={"Creator": "Experiment Microscope"},
    )
    return path


def annotate_provenance(ax, text: str) -> None:
    """A small caption in the figure corner recording where the data came from."""
    ax.figure.text(
        0.01, 0.01, text, fontsize=6, color="0.4", ha="left", va="bottom", wrap=True
    )


class SupportsExport:
    """Marker + default for views. ``export_figure`` returns the written path."""

    def can_export(self) -> bool:  # pragma: no cover - overridden
        return False

    def export_figure(self, path: str | Path, **opts: Any) -> Path:  # pragma: no cover
        raise NotImplementedError
