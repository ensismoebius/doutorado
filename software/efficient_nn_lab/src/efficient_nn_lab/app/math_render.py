"""Render the demos' plain-text equations as real mathematics.

The demo layers store every formula as a short ASCII/unicode string, e.g.
``"L = 1/2 (y - target)^2"`` or ``"dL/dz = dL/dy · σ'(z)"``. That is fine
for data, but the UI needs them typeset *mathematically* (fractions, real
sub/superscripts, sigma/sum glyphs) instead of shown as a code line.

This module bridges the gap:

* :func:`latexize` translates the pseudo-LaTeX the demos use into
  matplotlib's mathtext dialect -- the same engine every widget chart uses,
  so no external LaTeX distribution is needed, just the DejaVu fonts that
  ship with matplotlib.
* :func:`render_math_image` renders a translated expression to a
  transparent ``QImage`` (cached by expression/dpi/color).
* :class:`MathTextLabel` is the explanation widget: it turns ``$...$``
  fragments inside a prose string into highlighted inline equations while
  :meth:`MathTextLabel.text` keeps returning the original plain string, so
  existing callers/tests that compare against the raw explanation keep
  working unchanged.
"""

from __future__ import annotations

import base64
import html
import io
import re
from functools import lru_cache

from matplotlib import mathtext
from PySide6.QtCore import QByteArray, QBuffer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel

from efficient_nn_lab.app.theme import ACCENT_COLOR, TEXT_COLOR

# Inline equations inside the explanation panel are rendered at this DPI
# (~28 px cap height, one line) and tinted with the theme's highlight
# color so they stand out from the surrounding prose.
_INLINE_DPI = 190
_INLINE_COLOR = ACCENT_COLOR
_INLINE_MAX_WIDTH = 460

_MATH_DELIMITER_RE = re.compile(r"\$([^$]+)\$")

# Latin-1 letters (a-z, A-Z and all accented forms) used by every
# identifier-substitution regex below. `_UNICODE_LETTER` ranges cover
# U+00C0-U+00FF minus the two empty slots (× and ÷).
_UNICODE_LETTER = "A-Za-zÀ-ÖØ-öø-ÿ"


# --------------------------------------------------------------------------
# pseudo-LaTeX -> mathtext translation
# --------------------------------------------------------------------------

_WORD_GREEK = {
    "sigma": "\\sigma",
    "Sigma": "\\Sigma",
    "sum": "\\sum",
    "tau": "\\tau",
    "eta": "\\eta",
    "lambda": "\\lambda",
    "omega": "\\omega",
    "mu": "\\mu",
    "phi": "\\phi",
    "psi": "\\psi",
    "pi": "\\pi",
    "alpha": "\\alpha",
    "beta": "\\beta",
    "delta": "\\delta",
    "gamma": "\\gamma",
    "theta": "\\theta",
}

_WORD_GREEK_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in _WORD_GREEK) + r")\b")

# Unicode math symbols the demos write directly (σ, Σ, τ, ·, →, ...).
_CHAR_MAP = {
    "σ": "\\sigma",
    "Σ": "\\Sigma",
    "τ": "\\tau",
    "η": "\\eta",
    "λ": "\\lambda",
    "ω": "\\omega",
    "θ": "\\theta",
    "Δ": "\\Delta",
    "·": "\\cdot ",
    "×": "\\times",
    "→": "\\to",
    "←": "\\leftarrow",
    "↔": "\\leftrightarrow",
    "≈": "\\approx",
    "±": "\\pm",
    "∈": "\\in",
    "≠": "\\neq",
    "≤": "\\leq",
    "≥": "\\geq",
    "∞": "\\infty",
    "½": "\\tfrac{1}{2}",
}

_ASCII_SUBS = [
    (re.compile(r"<-"), "\\\\leftarrow "),
    (re.compile(r"->"), "\\\\to "),
    (re.compile(r"~="), "\\\\approx "),
    (re.compile(r">="), "\\\\geq "),
    (re.compile(r"<="), "\\\\leq "),
    (re.compile(r"\+-"), "\\\\pm "),
    (re.compile(r"(?<!\*)\*(?!\*)"), "\\\\cdot "),
    (re.compile(r" \. "), " \\\\cdot "),
]

# `a/b` -> \frac{a}{b}. Numerators/denominators may be a parenthesized
# group (``1 / (1 + e^-z)``) or a bare token; a denominator may carry a
# trailing argument ``dQ(w)``. Applied last, after all token rewrites, so
# it never trips over the backslashes those introduce.
_FRACTION_RE = re.compile(
    r"((?:\([^()]*\)|[^\s()/]+))\s*/\s*((?:\([^()]*\)|[^\s()/]+(?:\([^()]*\))?))"
)

# Derivative notations like ``dL/dy``, ``dL/dz_O``, ``dw/dw`` must NOT be
# turned into \frac{…}{…}.  We protect them with placeholders before the
# fraction regex and restore afterwards.
_DERIVATIVE_RE = re.compile(r"\b(d[A-Z])/(d[a-zA-Z](?:_\{[^}]*\})?)")

# ``\text{…}`` carries normal text with spaces inside math mode.
# Must be extracted before every other regex so the braces, spaces and
# letters are never consumed by subscripts, fractions, etc.
_TEXT_RE = re.compile(r"\\text\{([^}]*)\}")


def latexize(equation: str) -> str:
    """Translate a demo equation string into matplotlib mathtext.

    Inputs are the plain pseudo-LaTeX the demos use: ``"L = 1/2 (y -
    target)^2"``, ``"w <- w - eta . dL/dw"``, ``"z = w1·x1 + w2·x2"``.
    The output is a ``$``-free mathtext expression (callers wrap it).
    """
    s = equation

    # Protect \text{...} blocks first — they contain spaces and letters
    # that every later regex would mangle.
    _txt: list[str] = []
    def _txt_protect(m):
        _txt.append(m.group(0))
        return f"\x00§{len(_txt) - 1}§\x00"
    s = _TEXT_RE.sub(_txt_protect, s)

    s = _WORD_GREEK_RE.sub(lambda m: _WORD_GREEK[m.group(1)], s)
    s = re.sub(r"sum_", "\\\\sum_", s)
    s = re.sub(rf"([{_UNICODE_LETTER}])_([{_UNICODE_LETTER}0-9→←]+)", r"\1_{\2}", s)
    s = re.sub(rf"([{_UNICODE_LETTER}])([0-9]+)", r"\1_{\2}", s)
    s = re.sub(rf"\^(-?[{_UNICODE_LETTER}0-9])", r"^{\1}", s)
    for pattern, repl in _ASCII_SUBS:
        s = pattern.sub(repl, s)
    for ch, latex in _CHAR_MAP.items():
        s = s.replace(ch, latex)

    # Protect derivative notations (dL/dy, dL/dz_O, ...) and convert to \dfrac.
    _deriv: list[tuple[str, str]] = []
    def _deriv_protect(m):
        _deriv.append((m.group(1), m.group(2)))
        return f"\x00DERIV{len(_deriv) - 1}\x00"
    s = _DERIVATIVE_RE.sub(_deriv_protect, s)

    s = _FRACTION_RE.sub(lambda m: f"\\dfrac{{{m.group(1)}}}{{{m.group(2)}}}", s)

    for i, (num, den) in enumerate(_deriv):
        s = s.replace(f"\x00DERIV{i}\x00", f"\\dfrac{{{num}}}{{{den}}}")

    for i, orig in enumerate(_txt):
        s = s.replace(f"\x00§{i}§\x00", orig)

    return s


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _render_png(latex_expr: str, dpi: int, color: str) -> bytes:
    buf = io.BytesIO()
    mathtext.math_to_image(f"${latex_expr}$", buf, dpi=dpi, format="png", color=color)
    return buf.getvalue()


def render_math_image(expr: str, dpi: int = _INLINE_DPI, color: str = TEXT_COLOR) -> QImage | None:
    """Render one equation string to a transparent QImage.

    ``expr`` is translated with :func:`latexize`; on any parse failure the
    raw string is tried as-is, and if that also fails ``None`` is returned
    so callers can fall back to plain text.
    """
    latex = latexize(expr)
    for attempt in (latex, expr):
        try:
            data = _render_png(attempt, dpi, color)
        except Exception:
            continue
        if data:
            return QImage.fromData(data)
    return None


# --------------------------------------------------------------------------
# explanation widget
# --------------------------------------------------------------------------


def _qimage_to_png_b64(img: QImage) -> str:
    """Encode a QImage as a base64 PNG string."""
    buf = QByteArray()
    buffer = QBuffer(buf)
    buffer.open(QBuffer.WriteOnly)
    img.save(buffer, "PNG")
    return base64.b64encode(buf.data()).decode()


def _explanation_to_html(text: str) -> tuple[str, dict[str, QImage]]:
    """Split ``text`` on ``$...$`` and render each fragment as an image.

    Returns the rich-text body (with inline base64 data URIs) and an
    empty images dict — kept for API compatibility with callers that
    unpack the second element.
    """
    parts: list[str] = []
    index = 0
    for m in _MATH_DELIMITER_RE.finditer(text):
        if m.start() > index:
            parts.append(html.escape(text[index : m.start()], quote=False))
        fragment = m.group(1).strip()
        img = render_math_image(fragment, dpi=_INLINE_DPI, color=_INLINE_COLOR)
        if img is not None and not img.isNull() and img.width() > 0:
            if img.width() > _INLINE_MAX_WIDTH:
                img = img.scaledToWidth(_INLINE_MAX_WIDTH)
            b64 = _qimage_to_png_b64(img)
            parts.append(
                f'<img src="data:image/png;base64,{b64}"'
                f' align="absmiddle" style="margin: 0 2px;"/>'
            )
        else:
            parts.append(html.escape(m.group(0), quote=False))
        index = m.end()
    if index < len(text):
        parts.append(html.escape(text[index:], quote=False))
    return "".join(parts).replace("\n", "<br/>"), {}


class MathTextLabel(QLabel):
    """QLabel that renders ``$...$`` fragments as highlighted inline math.

    Images are embedded as base64 data URIs so no ``loadResource``
    override is needed.  :meth:`text` returns the *original* plain string
    (delimiters and all), so callers that compare against the raw
    explanation keep working.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        self._plain = ""
        super().__init__("", parent)
        if text:
            self.set_math_text(text)

    def text(self) -> str:
        return self._plain

    def set_math_text(self, text: str) -> None:
        self._plain = text or ""
        if "$" not in self._plain:
            super().setText(self._plain)
            return
        body, _ = _explanation_to_html(self._plain)
        super().setText(body if body else self._plain)
