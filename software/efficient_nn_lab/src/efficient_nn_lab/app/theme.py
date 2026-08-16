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

STYLESHEET = f"""
QMainWindow {{
    background: {BACKGROUND};
}}
QWidget {{
    color: {TEXT_COLOR};
    font-family: "DejaVu Sans";
    font-size: 18pt;
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
    font-size: 22pt;
    font-weight: 700;
    color: {TEXT_COLOR};
}}
QLabel#DemoTitle {{
    font-size: 24pt;
    font-weight: 800;
    color: {BITNET_COLOR};
}}
QLabel#Explanation {{
    color: #262B33;
}}
QLabel#Equation {{
    font-family: monospace;
    color: #444444;
}}
QLabel#Detail {{
    font-family: monospace;
    font-size: 15pt;
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
