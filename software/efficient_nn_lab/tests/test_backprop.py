import numpy as np
import pytest

from efficient_nn_lab.backprop.demos.traditional_gd import TraditionalBackpropDemo


def test_pipeline_checkpoints_match_hand_worked_numbers():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_pipeline"]
    last = checkpoints[-1].values
    assert last["x"] == pytest.approx(2.0)
    assert last["w"] == pytest.approx(0.1)
    assert last["y"] == pytest.approx(0.2)
    assert last["loss"] == pytest.approx(0.5 * (0.2 - 6.0) ** 2)
    assert last["grad_y"] == pytest.approx(0.2 - 6.0)
    assert last["grad_w"] == pytest.approx((0.2 - 6.0) * 2.0)
    assert last["w_updated"] == pytest.approx(0.1 - 0.1 * (0.2 - 6.0) * 2.0)


def test_gradient_descent_actually_converges_to_target():
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_convergence"]
    diffs = [abs(float(f.values["y"][-1]) - demo.target) for f in checkpoints]
    # monotonically shrinking distance to the target, ending very close.
    assert all(a >= b - 1e-9 for a, b in zip(diffs, diffs[1:]))
    assert diffs[0] > 5.0
    assert diffs[-1] < 0.1


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
