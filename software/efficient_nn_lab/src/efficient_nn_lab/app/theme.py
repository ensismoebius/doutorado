"""Shared colors and Qt stylesheet.

Colors mirror the semantic roles used in the LaTeX slide deck
(documentation/08-lectures/fronteiras-bitnets-redes-pulso) so the software
and the slides read as one visual system during the talk: blue for BitNet,
vermillion/red for SNN, green for "where they meet", amber for highlights,
grey for neutral chrome. Still a colorblind-safe (Okabe-Ito-derived)
palette, just pushed brighter/more saturated than the original muted set,
and paired with a darker neutral + higher fill opacity in the drawing
primitives (neuron_view/weight_view/signal_view) so shapes read as
confidently colored instead of washed out.

The stylesheet below pins Qt's font to "DejaVu Sans" explicitly, matching
matplotlib's own default font (used by every widgets/*.py chart without
any rcParams override) and the slide deck's \\setmainfont (see
documentation/08-lectures/fronteiras-bitnets-redes-pulso/preamble.tex) --
one font across Qt chrome, matplotlib panels, and the LaTeX slides.
"""

from __future__ import annotations


# Each hue is darkened just enough (WCAG relative-luminance formula) to
# clear 4.5:1 contrast against a *white* background -- these colors are
# used dually as fills/backgrounds (paired with their own dark or white
# text, not contrast-critical there) AND directly as small chart text/
# thin lines drawn on white matplotlib panels (contrast-critical, and the
# majority use case by call-site count). Since contrast is symmetric,
# a color that is dark enough for white text on top of it also reads
# clearly as text on white -- one value serves both roles. Amber/yellow
# pays for this the most: any yellow crossing 4.5:1 stops looking
# lemon-bright and reads as a deep gold/olive instead -- unavoidable,
# since pure yellow is inherently near-white in luminance.
BITNET_COLOR = "#0073E5"  # saturated blue, contrast 4.6:1 on white
SNN_COLOR = "#E61F00"  # vermillion-red, contrast 4.6:1 on white
CONVERGE_COLOR = "#008752"  # bluish green, contrast 4.6:1 on white
NEUTRAL_COLOR = "#5B6472"  # darker grey -- was too light to contrast against white
ACCENT_COLOR = "#966E00"  # deep amber/gold, contrast 4.6:1 on white

BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#EDF1F8"  # cooler, slightly more saturated than pure grey
TEXT_COLOR = "#111318"

# --------------------------------------------------------------------------
# Resolution-proportional UI scale ("like the games do"): one scale factor,
# computed from the window's current size against a reference resolution,
# multiplies every font size (and a couple of fixed-pixel chrome boxes in
# main_window.py) so the UI is legible and well-proportioned from a small
# 1024x768 window up through a maximized high-resolution one, instead of
# a single point size baked in for whatever screen it was designed on.
#
# These are the BASE sizes at scale == 1.0 (i.e. at exactly the reference
# resolution). Previously they were the literal, unscaled stylesheet values
# (18/22/24/14/15pt) — those were sized for a much bigger reference display
# than this app's actual minimum (900x600) and stated target floor
# (1024x768), which is what made every label look oversized there. Trimmed
# down here to normal desktop-UI proportions; the scale factor then adapts
# them to whatever window size is actually in use instead of leaving them
# fixed at one size for every resolution.
BASE_FONT_PT = 10
BASE_FRAME_TITLE_PT = 13
BASE_DEMO_TITLE_PT = 15
BASE_EQUATION_PT = 11
BASE_DETAIL_PT = 10

#: Smallest window this app is designed to run well in (matches the
#: usability floor documented in main_window.py's setMinimumSize comment,
#: rounded up to the stated target: "well fitted starting from 1024x768").
#: Scale == 1.0 here; windows below it still work (MainWindow's own
#: setMinimumSize is 900x600) but get a somewhat smaller, clamped scale.
REFERENCE_WIDTH = 1024
REFERENCE_HEIGHT = 768

#: Clamp range for the computed scale. The floor keeps text legible even at
#: the 900x600 absolute minimum window size; the ceiling stops a maximized
#: window on a large/4K display from blowing fonts up past what is still
#: useful for a demo that is mostly animation, not prose — games commonly
#: cap "UI scale" the same way rather than scaling it linearly to 4K.
MIN_UI_SCALE = 0.85
MAX_UI_SCALE = 1.5

#: Scale is quantized to this step so a live window-resize drag does not
#: reapply the stylesheet (and retrigger every dependent layout
#: measurement) on every single pixel of movement — only when the scale
#: actually crosses a visible increment.
_SCALE_STEP = 0.05


def compute_ui_scale(width: int, height: int) -> float:
    """Scale factor for a window of ``width`` x ``height``.

    ``min()`` of the two axis ratios (not e.g. the average) so neither
    dimension of the reference resolution is exceeded — a window that is
    wide but short scales by its height ratio, and vice versa, the same
    way a game fits its UI to the tighter of the two axes rather than
    overflowing one of them.
    """
    if width <= 0 or height <= 0:
        return MIN_UI_SCALE
    raw = min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)
    clamped = max(MIN_UI_SCALE, min(MAX_UI_SCALE, raw))
    return round(clamped / _SCALE_STEP) * _SCALE_STEP


def _scaled_pt(base_pt: int, scale: float) -> int:
    """Base point size times scale, floored so text never becomes unreadable."""
    return max(7, round(base_pt * scale))


def build_stylesheet(scale: float = 1.0) -> str:
    """The app's Qt stylesheet at the given UI scale (see ``compute_ui_scale``).

    Every font-size below is ``BASE_*_PT`` times ``scale`` — colors, borders
    and padding are deliberately left unscaled (a 1px border does not need
    to become 1.5px to stay legible the way text does, and scaling padding
    too would start fighting the measured-reservation layout in
    main_window.py, which already adapts to whatever the real font metrics
    report).
    """
    body_pt = _scaled_pt(BASE_FONT_PT, scale)
    frame_title_pt = _scaled_pt(BASE_FRAME_TITLE_PT, scale)
    demo_title_pt = _scaled_pt(BASE_DEMO_TITLE_PT, scale)
    equation_pt = _scaled_pt(BASE_EQUATION_PT, scale)
    detail_pt = _scaled_pt(BASE_DETAIL_PT, scale)
    return f"""
QMainWindow {{
    background: {BACKGROUND};
}}
QWidget {{
    color: {TEXT_COLOR};
    font-family: "DejaVu Sans";
    font-size: {body_pt}pt;
}}
QListWidget, QTreeWidget {{
    background: {PANEL_BACKGROUND};
    border: 1px solid #B6C0D6;
}}
QTextEdit {{
    background: {BACKGROUND};
    color: {TEXT_COLOR};
    border: 1px solid #B6C0D6;
    padding: 8px;
}}
QTreeWidget::item {{
    padding: 3px 2px;
}}
QTreeWidget::item:hover {{
    background: #DCE6FA;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {BITNET_COLOR};
    color: white;
    font-weight: 700;
}}
QLabel#FrameTitle {{
    font-size: {frame_title_pt}pt;
    font-weight: 700;
    color: {TEXT_COLOR};
}}
QLabel#DemoTitle {{
    font-size: {demo_title_pt}pt;
    font-weight: 800;
    color: {BITNET_COLOR};
}}
QLabel#Explanation {{
    color: #262B33;
}}
QFrame#EquationFrame {{
    background: #F4F7FC;
    border: 1px solid #B6C0D6;
    border-radius: 6px;
}}
QLabel#Equation {{
    color: #666666;
    font-size: {equation_pt}pt;
    font-style: italic;
}}
QLabel#Detail {{
    font-family: monospace;
    font-size: {detail_pt}pt;
    color: #555555;
}}
QPushButton {{
    padding: 6px 14px;
    background: {PANEL_BACKGROUND};
    border: 1px solid #A6B2C9;
    border-radius: 4px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: #D6E2F7;
    border-color: {BITNET_COLOR};
}}
QPushButton:pressed {{
    background: {BITNET_COLOR};
    color: white;
    border-color: {BITNET_COLOR};
}}
QPushButton:checked {{
    background: {ACCENT_COLOR};
    color: white;
    border-color: {ACCENT_COLOR};
}}
QPushButton:disabled {{
    color: #9AA1AC;
    background: #ECEEF1;
}}
"""


#: Back-compat default (scale == 1.0) for any caller that just wants "the"
#: stylesheet without computing a window-specific scale.
STYLESHEET = build_stylesheet(1.0)
