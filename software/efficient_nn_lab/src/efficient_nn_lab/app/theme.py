"""Shared colors and Qt stylesheet.

Colors intentionally mirror the semantic roles used in the LaTeX slide deck
(documentation/08-lectures/fronteiras-bitnets-redes-pulso) so the software
and the slides read as one visual system during the talk: blue for BitNet,
vermillion for SNN, green for "where they meet", grey for neutral chrome.
Same Okabe-Ito-derived palette, so it stays colorblind-safe too.
"""

from __future__ import annotations

BITNET_COLOR = "#0072B2"  # Okabe-Ito blue
SNN_COLOR = "#D55E00"  # Okabe-Ito vermillion
CONVERGE_COLOR = "#009E73"  # Okabe-Ito bluish green
NEUTRAL_COLOR = "#A6A6A6"
ACCENT_COLOR = "#E69F00"  # Okabe-Ito orange, used sparingly for highlights

BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#1A1A1A"

STYLESHEET = f"""
QMainWindow {{
    background: {BACKGROUND};
}}
QWidget {{
    color: {TEXT_COLOR};
    font-size: 11pt;
}}
QListWidget {{
    background: #FAFAFA;
    border: 1px solid #DDDDDD;
}}
QListWidget::item:selected {{
    background: {BITNET_COLOR};
    color: white;
}}
QLabel#FrameTitle {{
    font-size: 14pt;
    font-weight: 600;
}}
QLabel#DemoTitle {{
    font-size: 16pt;
    font-weight: 700;
    color: {BITNET_COLOR};
}}
QLabel#Explanation {{
    color: #333333;
}}
QPushButton {{
    padding: 6px 12px;
}}
QPushButton:checked {{
    background: {ACCENT_COLOR};
}}
"""
