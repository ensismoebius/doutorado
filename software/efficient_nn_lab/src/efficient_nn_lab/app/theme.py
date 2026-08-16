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

BITNET_COLOR = "#0090FF"  # bright, saturated blue
SNN_COLOR = "#FF5A36"  # vivid vermillion-red
CONVERGE_COLOR = "#00C389"  # bright bluish green
NEUTRAL_COLOR = "#6B7280"  # darker grey -- was too light to contrast against white
ACCENT_COLOR = "#FFB000"  # vivid amber, used for highlights/glows

BACKGROUND = "#FFFFFF"
PANEL_BACKGROUND = "#F4F6F9"
TEXT_COLOR = "#111318"

STYLESHEET = f"""
QMainWindow {{
    background: {BACKGROUND};
}}
QWidget {{
    color: {TEXT_COLOR};
    font-family: "DejaVu Sans";
    font-size: 11pt;
}}
QListWidget, QTreeWidget {{
    background: {PANEL_BACKGROUND};
    border: 1px solid #C7CDD6;
}}
QTreeWidget::item {{
    padding: 3px 2px;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {BITNET_COLOR};
    color: white;
}}
QLabel#FrameTitle {{
    font-size: 14pt;
    font-weight: 700;
    color: {TEXT_COLOR};
}}
QLabel#DemoTitle {{
    font-size: 16pt;
    font-weight: 800;
    color: {BITNET_COLOR};
}}
QLabel#Explanation {{
    color: #262B33;
}}
QPushButton {{
    padding: 6px 14px;
    background: {PANEL_BACKGROUND};
    border: 1px solid #B7BFCB;
    border-radius: 4px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: #E4E9F0;
}}
QPushButton:pressed {{
    background: {BITNET_COLOR};
    color: white;
}}
QPushButton:checked {{
    background: {ACCENT_COLOR};
    color: #1A1200;
    border-color: {ACCENT_COLOR};
}}
QPushButton:disabled {{
    color: #9AA1AC;
    background: #ECEEF1;
}}
"""
