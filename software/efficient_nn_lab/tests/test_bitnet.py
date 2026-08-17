import numpy as np
import pytest

from efficient_nn_lab.bitnet.linear import (
    loss_gradient_wrt_y,
    quantized_forward,
    squared_error_loss,
)
from efficient_nn_lab.bitnet.quantization import ternary_quantize, ternary_quantize_np
from efficient_nn_lab.bitnet.ste import sgd_update, ste_backward, ste_forward
from efficient_nn_lab.bitnet.demos.backward import BackwardSTEDemo
from efficient_nn_lab.bitnet.demos.forward import ForwardLossDemo
from efficient_nn_lab.bitnet.demos.guided_sequence import GuidedBitNetDemo
from efficient_nn_lab.bitnet.demos.scalar_quantization import ScalarQuantizationDemo


# -- quantization -------------------------------------------------------
def test_ternary_quantization():
    assert ternary_quantize(-0.8) == -1
    assert ternary_quantize(-0.2) == 0
    assert ternary_quantize(0.2) == 0
    assert ternary_quantize(0.8) == 1


def test_ternary_quantization_threshold_boundary():
    # exactly at threshold falls into the dead zone (strict inequality)
    assert ternary_quantize(0.5, threshold=0.5) == 0
    assert ternary_quantize(-0.5, threshold=0.5) == 0
    assert ternary_quantize(0.50001, threshold=0.5) == 1
    assert ternary_quantize(-0.50001, threshold=0.5) == -1


def test_ternary_quantize_np_matches_scalar():
    w = np.array([-0.8, -0.2, 0.2, 0.8])
    expected = [ternary_quantize(float(v)) for v in w]
    np.testing.assert_array_equal(ternary_quantize_np(w), expected)


# -- STE ------------------------------------------------------------------
def test_ste_forward_matches_quantization():
    assert ste_forward(0.8) == ternary_quantize(0.8)


def test_ste_backward_is_identity():
    assert ste_backward(1.23) == 1.23
    assert ste_backward(-4.0) == -4.0


def test_sgd_update():
    assert sgd_update(0.80, -4.0, 0.01) == pytest.approx(0.84)


# -- linear neuron / loss (spec §9, §10 worked example) -----------------
def test_quantized_forward_matches_spec_example():
    result = quantized_forward((2.0, 3.0), (0.8, 0.2))
    assert result.w_quant == (1, 0)
    assert result.y == pytest.approx(2.0)


def test_squared_error_loss_matches_spec_example():
    assert squared_error_loss(y=2.0, target=4.0) == pytest.approx(2.0)


def test_loss_gradient_wrt_y():
    assert loss_gradient_wrt_y(y=2.0, target=4.0) == pytest.approx(-2.0)


# -- demo modules: numeric correctness ------------------------------------
def test_guided_sequence_matches_spec_numbers():
    demo = GuidedBitNetDemo()
    checkpoints = demo.checkpoint_frames()
    assert len(checkpoints) == 10
    by_step = {int(f.values["step_number"]): f.values for f in checkpoints}

    assert by_step[1]["w_value"] == pytest.approx(0.80)
    assert by_step[2]["q_value"] == 1
    assert by_step[4]["y_value"] == pytest.approx(2.0)
    assert by_step[6]["loss_value"] == pytest.approx(2.0)
    assert by_step[9]["w_value"] == pytest.approx(0.84)
    # the representation used in the forward pass does not change here
    assert by_step[10]["q_value"] == 1


def test_forward_loss_demo_default_matches_spec():
    demo = ForwardLossDemo()
    loss_frame = demo.checkpoint_frames()[-1]
    assert loss_frame.values["loss"] == pytest.approx(2.0)
    assert loss_frame.values["loss_reveal"] == pytest.approx(1.0)


def test_scalar_quantization_demo_slides_to_quantized_level():
    demo = ScalarQuantizationDemo()
    checkpoints = demo.checkpoint_frames()
    first, last = checkpoints[0], checkpoints[-1]
    assert first.values["w_display"] == pytest.approx(0.80)
    assert last.values["w_quant"] == 1
    assert last.values["w_display"] == pytest.approx(1.0)


def test_scalar_quantization_slide_is_smooth_not_instant():
    # the whole point of the redesign: many frames between the two
    # checkpoints, with the displayed value moving monotonically from
    # 0.80 up to the quantized level 1.0 rather than jumping.
    demo = ScalarQuantizationDemo()
    assert len(demo._frames) > 10
    values = [f.values["w_display"] for f in demo._frames]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))
    assert values[0] == pytest.approx(0.80)
    assert values[-1] == pytest.approx(1.0)


def test_backward_demo_exposes_both_ste_paths():
    demo = BackwardSTEDemo()
    checkpoints = demo.checkpoint_frames()
    kinds = {f.values["kind"] for f in checkpoints}
    assert kinds == {"staircase", "quant_derivative", "ste_pipeline"}
    ste_frames = [f for f in checkpoints if f.values["kind"] == "ste_pipeline"]
    reveals = [(f.values["fwd_reveal"], f.values["bwd_reveal"], f.values["joined_reveal"]) for f in ste_frames]
    assert reveals[0] == (1.0, 0.0, 0.0)  # forward path only
    assert reveals[1] == (1.0, 1.0, 0.0)  # backward path joins in
    assert reveals[2] == (1.0, 1.0, 1.0)  # both paths, joined


def test_backward_demo_worked_example_is_hand_computable():
    # default w = 0.65, tau = 0.5: Q(w) = +1, y = x*Q(w) = 2*1 = 2,
    # L = 1/2 (2 - 4)^2 = 2, ∂L/∂Q(w) = 2 - 4 = -2. The STE substitute for
    # dQ/dw is 1, so ∂L/∂w_ste = -2 while ∂L/∂w_real = 0.
    demo = BackwardSTEDemo()
    ste = [f for f in demo.checkpoint_frames() if f.values["kind"] == "ste_pipeline"]
    assert ste
    v = ste[0].values
    assert v["w"] == pytest.approx(0.65)
    assert v["w_quant"] == 1
    assert v["y"] == pytest.approx(2.0)
    assert v["loss"] == pytest.approx(2.0)
    assert v["upstream_grad"] == pytest.approx(-2.0)  # ∂L/∂Q(w)
    assert v["dq_dw_real"] == pytest.approx(0.0)
    assert v["dq_dw_ste"] == pytest.approx(1.0)
    assert v["dl_dw_real"] == pytest.approx(0.0)
    assert v["dl_dw_ste"] == pytest.approx(-2.0)
    assert v["threshold"] == pytest.approx(0.5)


def test_backward_demo_all_scenes_share_the_same_example_weight():
    # every checkpoint of every kind carries the same concrete example w,
    # so the staircase, the derivative graph and the block diagram all
    # talk about the same worked example.
    demo = BackwardSTEDemo()
    example_ws = {
        float(f.values["example_w"]) for f in demo.checkpoint_frames() if "example_w" in f.values
    }
    assert example_ws == {0.65}


def test_backward_demo_parameter_change_recomputes_the_example():
    demo = BackwardSTEDemo()
    demo.set_parameter("w", -0.3)
    ste = [f for f in demo.checkpoint_frames() if f.values["kind"] == "ste_pipeline"]
    assert ste
    v = ste[0].values
    assert v["w"] == pytest.approx(-0.3)
    assert v["w_quant"] == 0  # inside the dead zone -> Q(w) = 0
    assert v["y"] == pytest.approx(0.0)
    assert v["dl_dw_real"] == pytest.approx(0.0)
    assert v["dl_dw_ste"] == v["upstream_grad"]  # STE passes it through


def test_backward_demo_derivative_graph_morphs_zero_into_ste_constant():
    # the real derivative (zero everywhere) actually tweens into the
    # constant-1 curve the STE substitutes -- not just two static pictures.
    demo = BackwardSTEDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "quant_derivative"]
    assert len(checkpoints) == 2
    real, ste = checkpoints
    assert np.all(real.values["curve"] == 0.0)
    assert np.all(ste.values["curve"] == 1.0)
    mid_index = demo._checkpoint_frame_indices[demo.checkpoint_frames().index(ste)] - 3
    mid_tween = demo._frames[mid_index]
    assert 0.0 < mid_tween.values["curve"].max() < 1.0


def test_backward_demo_kind_changes_only_at_deliberate_cuts():
    # deliberate scene changes (different widget/kind): no tween frames
    # blending two structurally unrelated pictures. staircase -> derivative
    # graph -> block diagram is two genuine cuts, not one.
    demo = BackwardSTEDemo()
    kinds_in_order = [f.values["kind"] for f in demo._frames]
    changes = sum(1 for a, b in zip(kinds_in_order, kinds_in_order[1:]) if a != b)
    assert changes == 2
