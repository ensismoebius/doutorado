"""Math rendering pipeline: latexize translation, render_math_image, and the
MathTextLabel widget backing the equation panel and the explanation
highlighting.

The core guarantee the feature rests on: every ``$...$`` fragment the demos
mark up must translate to something matplotlib mathtext can typeset, and the
explanation label must keep exposing the raw plain text (with the ``$``
markers) so ``test_main_window``'s ``text() == frame.explanation`` contract
keeps holding.
"""

import re

import pytest

from efficient_nn_lab.app.main_window import _build_demo_tree
from efficient_nn_lab.app.math_render import (
    MathTextLabel,
    _explanation_to_html,
    latexize,
    render_math_image,
)


def _all_demos():
    demos = []
    for group in _build_demo_tree().values():
        demos.extend(group)
    return demos


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("w = 0.8", r"w = 0.8"),
        ("e^-z", r"e^{-z}"),
        ("x^2", r"x^{2}"),
        ("w_D→O", r"w_{D\toO}"),
        ("sum_i x_i . Q(w_i)", r"\sum_{i} x_{i} \cdot Q(w_{i})"),
        ("1 / (1 + e^-z)", r"\dfrac{1}{(1 + e^{-z})}"),
        ("sigma(z)", r"\sigma(z)"),
        ("tau", r"\tau"),
        ("V_th = 0.50", r"V_{th} = 0.50"),
        ("P(spike) = 0.80", r"P(spike) = 0.80"),
        ("diferença = y - target", r"diferença = y - target"),
    ],
)
def test_latexize_translations(raw, expected):
    assert latexize(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Q(w) = +1 se w > tau; -1 se w < -tau; 0 caso contrario",
        "y = sigma(z) = 1 / (1 + e^-z)",
        "dL/dw ~= (y - target) . x  [STE]",
        "sigmoide(v) = 0,5 + (v - v_th) / (1 + k|v - v_th|)",
        "L = 1/2 (y - target)^2",
        "w <- w - eta . dL/dw",
        "z = w1·y_C + w2·y_D;  y = σ(z)",
        "S(v) = 1 se v >= v_th; 0 caso contrario",
    ],
)
def test_latexize_accepts_real_equations(raw):
    latexize(raw)


def test_every_demo_explanation_fragment_translates(qapp):
    """All $...$ fragments the demos carry must be mathtext-translatable."""
    count = 0
    for demo in _all_demos():
        for frame in demo._frames:
            for match in re.finditer(r"\$([^$]+)\$", frame.explanation or ""):
                latexize(match.group(1))
                count += 1
    assert count > 100


def test_every_demo_equation_renders_to_image(qapp):
    for demo in _all_demos():
        for frame in demo._frames:
            if not frame.equation:
                continue
            img = render_math_image(frame.equation)
            assert img is not None, f"{demo.slug}: {frame.equation}"
            assert not img.isNull()


def test_explanation_to_html_embeds_images(qapp):
    body, images = _explanation_to_html("O alvo é $target = 4$; a saída é $y = 2$.")
    assert body.count('data:image/png;base64,') == 2
    assert images == {}
    assert "O alvo é " in body
    assert "target = 4" not in body  # fragment is replaced by the image


def test_explanation_to_html_escapes_prose(qapp):
    body, images = _explanation_to_html("a < b & c > d")
    assert images == {}
    assert "<b" not in body
    assert "&lt;" in body


def test_unclosed_dollar_stays_literal(qapp):
    label = MathTextLabel("taxa $ r_max")
    assert label.text() == "taxa $ r_max"


def test_math_text_round_trips_plain_string(qapp):
    label = MathTextLabel("x1=2")
    assert label.text() == "x1=2"
