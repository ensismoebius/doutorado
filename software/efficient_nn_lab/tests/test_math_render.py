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
        # the trailing space matters: "w_{D\toO}" is an unknown symbol and
        # used to be silently rendered as raw text (see the regression test
        # test_char_map_never_glues_a_command_to_the_next_letter below).
        ("w_D→O", r"w_{D\to O}"),
        ("sum_i x_i . Q(w_i)", r"\sum_{i} x_{i} \cdot Q(w_{i})"),
        ("1 / (1 + e^-z)", r"\dfrac{1}{(1 + e^{-z})}"),
        ("sigma(z)", r"\sigma(z)"),
        ("tau", r"\tau"),
        ("V_th = 0.50", r"V_{th} = 0.50"),
        ("P(spike) = 0.80", r"P(spike) = 0.80"),
        ("diferença = y - target", r"diferença = y - target"),
        # ½ must become \frac, not \tfrac: mathtext has no \tfrac, so the
        # old spelling parsed nowhere and every equation using ½ fell back
        # to raw pseudo-LaTeX on screen (found by backprop.chain's loss
        # equation, the first one to use the glyph).
        ("L = ½ (a_2 - alvo)^2", r"L = \frac{1}{2} (a_{2} - alvo)^{2}"),
        ("δ_2 = ∂L/∂z2", r"\delta _2 = \dfrac{\partial L}{\partial z_{2}}"),
    ],
)
def test_latexize_translations(raw, expected):
    assert latexize(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Q(w) = +1 \\text{ se: } w > tau; -1 \\text{ se: } w < -tau; 0 \\text{ caso contrario}",
        "y = sigma(z) = 1 / (1 + e^-z)",
        "∂L/∂w ~= (y - target) . x  [STE]",
        "sigmoide(v) = 0,5 + (v - v_th) / (1 + k|v - v_th|)",
        "L = 1/2 (y - target)^2",
        "w <- w - eta . ∂L/∂w",
        "z = w1·y_C + w2·y_D;  y = σ(z)",
        "S(v) = 1 \\text{ se: } v >= v_th; 0 \\text{ caso contrario}",
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


def test_every_demo_equation_is_really_parsed_as_math(qapp):
    """Not just "an image came back" -- the *translated* expression must parse.

    render_math_image falls back to rendering the raw string when mathtext
    rejects the translation, so the test above passes even when the audience
    is shown pseudo-LaTeX source. Four equations of the 4-layer demo shipped
    broken that way (``w_A→C`` translated to ``w_{A\toC}``, an unknown
    symbol) because nothing asserted the parse itself. This does.
    """
    from matplotlib import mathtext

    parser = mathtext.MathTextParser("agg")
    for demo in _all_demos():
        for equation in {f.equation for f in demo.checkpoint_frames() if f.equation}:
            translated = latexize(equation)
            try:
                parser.parse(f"${translated}$")
            except Exception as exc:  # noqa: BLE001 - report which one and why
                raise AssertionError(
                    f"{demo.slug}: {equation!r} -> {translated!r} rejeitada: {exc}"
                ) from None


@pytest.mark.parametrize(
    "raw, expected",
    [
        # partial derivatives (the notation the slides use for loss gradients)
        ("∂L/∂w", r"\dfrac{\partial L}{\partial w}"),
        ("∂L/∂z_O", r"\dfrac{\partial L}{\partial z_{O}}"),
        # total derivatives keep the plain d: single-variable functions
        # (the LIF membrane ODE, Q(w), S(v)) -- same split as the slides.
        ("dV/dt", r"\dfrac{dV}{dt}"),
        ("dQ/dw", r"\dfrac{dQ}{dw}"),
        ("dS/dv", r"\dfrac{dS}{dv}"),
    ],
)
def test_latexize_keeps_the_derivative_operator_the_author_chose(raw, expected):
    assert latexize(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "w_A→C",        # subscript ending in an arrow, then a letter
        "σ'(z_A)",      # greek then a quote
        "tau_rec",      # word-spelled greek followed by a subscript
        "a ≈ b",
        "x ∈ S",
    ],
)
def test_char_map_never_glues_a_command_to_the_next_letter(raw):
    """``→`` + ``C`` must not become ``\toC`` (an unknown symbol).

    Every control word _CHAR_MAP emits has to be terminated. This is the
    regression that made four of the 4-layer demo's equations unrenderable.
    """
    from matplotlib import mathtext

    mathtext.MathTextParser("agg").parse(f"${latexize(raw)}$")


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
