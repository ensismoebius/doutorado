"""Lock the surrogate-gradient worked example (ESPECIFICACAO_DLVL.md #20).

The demo docstring promises one concrete example that runs through every
scene: v_th = 1.0, v = 1.2 (0.2 above threshold), spike S(v) = 1,
sigmoid = 0.60, real derivative = 0, surrogate gradient = 0.25. These
same numbers are drawn as markers by the surrogate_curve widget, so if
either the code or the docstring drifts, these tests catch the mismatch.
"""

import pytest

from efficient_nn_lab.snn.demos.surrogate_gradient import SurrogateGradientDemo

K = 5.0
V_TH = 1.0
EXAMPLE_V = 1.2
EXAMPLE_VMT = 0.2
EXAMPLE_SPIKE = 1.0
EXAMPLE_SIGMOID = 0.60
EXAMPLE_SURROGATE = 0.25

REFERENCE = {
    "example_v": EXAMPLE_V,
    "example_vmt": EXAMPLE_VMT,
    "example_spike": EXAMPLE_SPIKE,
    "example_sigmoid": EXAMPLE_SIGMOID,
    "example_surrogate": EXAMPLE_SURROGATE,
}


def _surrogate_curve_checkpoints(demo):
    return [
        f.values
        for f in demo._frames
        if f.is_checkpoint and f.values.get("kind") == "surrogate_curve"
    ]


def test_docstring_still_promises_the_example_values():
    import re

    import efficient_nn_lab.snn.demos.surrogate_gradient as surrogate_module

    doc = re.sub(r"\s+", " ", surrogate_module.__doc__ or "")
    for literal in (
        "v_th = 1.0",
        "v = 1.2",
        "0.2 above the threshold",
        "S(v) = 1",
        "sigmoid = 0.60",
        "surrogate gradient = 0.25",
    ):
        assert literal in doc, f"docstring drifted: {literal!r} missing"


@pytest.mark.parametrize("key,expected", list(REFERENCE.items()))
def test_surrogate_curve_frames_carry_documented_markers(key, expected):
    demo = SurrogateGradientDemo()
    frames = _surrogate_curve_checkpoints(demo)
    assert frames, "no surrogate_curve checkpoint frames found"
    for values in frames:
        assert key in values, f"widget marker key {key!r} missing from frame"
        assert values[key] == pytest.approx(expected, abs=1e-6)


def test_widget_reads_exact_key_names():
    import inspect

    import efficient_nn_lab.widgets.weight_view as weight_view

    source = inspect.getsource(weight_view.WeightView._render_surrogate_curve)
    for key in REFERENCE:
        assert key in source, f"widget no longer reads marker key {key!r}"
