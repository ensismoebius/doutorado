import numpy as np
import pytest

from efficient_nn_lab.backprop.activation import sigmoid, sigmoid_derivative
from efficient_nn_lab.backprop.demos.traditional_gd import TraditionalBackpropDemo


def test_pipeline_checkpoints_match_hand_worked_numbers():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_pipeline"]
    last = checkpoints[-1].values
    x, w, target, lr = demo.x, demo.w_init, demo.target, demo.learning_rate
    z = w * x
    y = sigmoid(z)
    slope = sigmoid_derivative(z)
    grad_y = y - target
    grad_z = grad_y * slope
    grad_w = grad_z * x

    assert last["x"] == pytest.approx(x)
    assert last["w"] == pytest.approx(w)
    assert last["z"] == pytest.approx(z)
    assert last["y"] == pytest.approx(y)
    assert 0.0 < last["y"] < 1.0  # the whole point of the sigmoid: bounded output
    assert last["loss"] == pytest.approx(0.5 * (y - target) ** 2)
    assert last["grad_y"] == pytest.approx(grad_y)
    assert last["grad_z"] == pytest.approx(grad_z)
    assert last["grad_w"] == pytest.approx(grad_w)
    assert last["w_updated"] == pytest.approx(w - lr * grad_w)


def test_activation_step_is_a_distinct_checkpoint_from_the_linear_combination():
    # z = w*x and y = sigma(z) must be two separate, separately-revealed
    # steps -- not folded into one "forward" step the way the pre-sigmoid
    # version of this demo did it.
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_pipeline"]
    z_only = next(f for f in checkpoints if f.values["z_reveal"] >= 0.99 and f.values["y_reveal"] < 0.5)
    z_and_y = next(f for f in checkpoints if f.values["y_reveal"] >= 0.99)
    assert z_only.values["z"] == pytest.approx(demo.w_init * demo.x)
    assert z_and_y.values["y"] == pytest.approx(sigmoid(demo.w_init * demo.x))


def test_gradient_descent_actually_converges_to_target():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_convergence"]
    diffs = [abs(float(f.values["y"][-1]) - demo.target) for f in checkpoints]
    # monotonically shrinking distance to the target, ending very close.
    assert all(a >= b - 1e-9 for a, b in zip(diffs, diffs[1:]))
    assert diffs[0] > 0.5
    assert diffs[-1] < 0.05


def test_convergence_chart_output_stays_within_sigmoid_range():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_convergence"]
    for f in checkpoints:
        assert np.all(f.values["y"] > 0.0)
        assert np.all(f.values["y"] < 1.0)


def test_sigmoid_inset_fields_are_consistent_with_the_activation():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_convergence"]
    for f in checkpoints:
        z, slope = f.values["z"], f.values["slope"]
        assert slope == pytest.approx(sigmoid_derivative(z))
        assert f.values["y"][-1] == pytest.approx(sigmoid(z))


def test_convergence_chart_grows_by_one_point_per_checkpoint():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_convergence"]
    lengths = [len(f.values["y"]) for f in checkpoints]
    assert lengths == list(range(1, len(checkpoints) + 1))


def test_convergence_animation_slides_between_iterations_not_instant():
    # within one iteration's gap, the last (growing) point should move
    # smoothly rather than jump straight from the old value to the new one.
    demo = TraditionalBackpropDemo()
    convergence_checkpoint_indices = [
        i for i in demo._checkpoint_frame_indices if demo._frames[i].values["kind"] == "backprop_convergence"
    ]
    first, second = convergence_checkpoint_indices[0], convergence_checkpoint_indices[1]
    mid = demo._frames[(first + second) // 2]
    assert mid.values["kind"] == "backprop_convergence"
    y_start = demo._frames[first].values["y"][-1]
    y_end = demo._frames[second].values["y"][-1]
    y_mid = mid.values["y"][-1]
    lo, hi = sorted((float(y_start), float(y_end)))
    assert lo < float(y_mid) < hi


def test_pipeline_to_convergence_is_a_deliberate_cut():
    demo = TraditionalBackpropDemo()
    kinds_in_order = [f.values["kind"] for f in demo._frames]
    changes = sum(1 for a, b in zip(kinds_in_order, kinds_in_order[1:]) if a != b)
    assert changes == 1
