import numpy as np
import pytest

from efficient_nn_lab.backprop.activation import sigmoid, sigmoid_derivative
from efficient_nn_lab.backprop.demos.multilayer_network import MultilayerNetworkDemo
from efficient_nn_lab.backprop.demos.traditional_gd import TraditionalBackpropDemo


def test_pipeline_checkpoints_match_hand_worked_numbers():
    # only the first cycle uses w_init -- later cycles repeat the same 9
    # steps starting from whatever w the previous cycle's update produced
    # (see test_pipeline_cycle_repeats_until_close_enough_to_target).
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_pipeline"]
    last = checkpoints[8].values
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


def test_pipeline_cycle_repeats_until_close_enough_to_target():
    # phase 1 (the detailed 9-step walkthrough) must not stop after a
    # single, possibly-still-far-from-target pass -- it repeats the whole
    # cycle, each time starting from the previous cycle's updated w, until
    # the output is close enough to the target (mirrors phase 2's own
    # repeat-until-close-enough loop, see test_gradient_descent_actually_converges_to_target).
    demo = TraditionalBackpropDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["kind"] == "backprop_pipeline"]
    assert len(checkpoints) % 9 == 0
    n_cycles = len(checkpoints) // 9
    assert n_cycles > 1  # the whole point: more than one repetition happens

    iterations = [f.values["iteration"] for f in checkpoints]
    assert iterations == sorted(iterations)
    assert iterations[0] == 1
    assert iterations[-1] == n_cycles

    # each cycle's starting w is exactly the previous cycle's w_updated --
    # the repetition is a genuine continuation, not independent restarts.
    cycle_start_ws = [checkpoints[i * 9].values["w"] for i in range(n_cycles)]
    cycle_end_w_updates = [checkpoints[i * 9 + 8].values["w_updated"] for i in range(n_cycles)]
    for k in range(n_cycles - 1):
        assert cycle_start_ws[k + 1] == pytest.approx(cycle_end_w_updates[k])

    # the last cycle's final output is within the demo's convergence
    # tolerance of the target -- that is *why* the repetition stopped.
    last_y = checkpoints[-1].values["y"]
    assert abs(last_y - demo.target) < 0.05


def test_pipeline_to_convergence_is_a_deliberate_cut():
    demo = TraditionalBackpropDemo()
    kinds_in_order = [f.values["kind"] for f in demo._frames]
    changes = sum(1 for a, b in zip(kinds_in_order, kinds_in_order[1:]) if a != b)
    assert changes == 1


def test_pipeline_inset_is_held_during_tweens_and_snaps_at_checkpoints():
    # The sigmoid inset's activation point and derivative (tangent) must
    # only move when the step that updates them appears -- not glide toward
    # the next iteration during the interpolated transition between cycles.
    demo = TraditionalBackpropDemo()
    frames = demo._frames
    pipeline = [i for i, f in enumerate(frames) if f.values["kind"] == "backprop_pipeline"]

    # pick the boundary tween between cycle 1's last checkpoint and cycle
    # 2's first checkpoint.
    first2 = next(
        i for i in pipeline
        if frames[i].is_checkpoint and frames[i].label == "Iteração 2 — O neurônio"
    )
    prev_idx = next(
        i for i in reversed(pipeline)
        if i < first2 and frames[i].is_checkpoint
    )
    prev = frames[prev_idx].values
    tweens = [frames[i].values for i in range(prev_idx + 1, first2)]
    assert tweens and all(not frames[i].is_checkpoint for i in range(prev_idx + 1, first2))

    for key in ("z", "y", "slope", "grad_y", "grad_z"):
        assert all(f[key] == pytest.approx(prev[key]) for f in tweens), key

    # the destination checkpoint does carry the new iteration's values --
    # that is the step at which the inset is finally allowed to move.
    arrived = frames[first2].values
    assert arrived["z"] != pytest.approx(prev["z"])


def test_convergence_inset_is_frozen_during_substeps_and_moves_at_iterations():
    # In the convergence phase each iteration interpolates _SUBSTEPS_PER_
    # ITERATION frames between the previous and next w/z/loss, but the
    # sigmoid inset (activation point + derivative) must stay at the last
    # completed iteration the whole time and only jump when the iteration
    # checkpoint itself lands.
    demo = TraditionalBackpropDemo()
    last_checkpoint_z = None
    substeps = 0
    for f in demo._frames:
        if f.values.get("kind") != "backprop_convergence":
            continue
        z, slope, point_y = f.values["z"], f.values["slope"], f.values["point_y"]
        trail = np.asarray(f.values["z_trail"])
        if f.is_checkpoint:
            last_checkpoint_z = z
            assert point_y == pytest.approx(sigmoid(z))
        else:
            assert last_checkpoint_z is not None
            assert z == pytest.approx(last_checkpoint_z)
            assert slope == pytest.approx(sigmoid_derivative(last_checkpoint_z))
            assert point_y == pytest.approx(sigmoid(last_checkpoint_z))
            assert trail[-1] == pytest.approx(last_checkpoint_z)
            substeps += 1
    assert substeps > 0


# -- MultilayerNetworkDemo (3 -> 2 -> 2 -> 1, all sigmoid) -------------------

def _manual_mlp_forward_backward(demo: MultilayerNetworkDemo):
    x = np.array([1.0, 0.5, -0.5])
    w1 = np.array([[0.4, -0.3, 0.6], [-0.2, 0.5, 0.1]])
    w2 = np.array([[0.7, -0.4], [0.3, 0.6]])
    w3 = np.array([[0.5, -0.6]])

    z1 = w1 @ x
    y1 = sigmoid(z1)
    z2 = w2 @ y1
    y2 = sigmoid(z2)
    zO = float((w3 @ y2)[0])
    yO = float(sigmoid(zO))

    grad_yO = yO - demo.target
    grad_zO = grad_yO * sigmoid_derivative(zO)
    grad_y2 = w3[0] * grad_zO
    grad_z2 = grad_y2 * sigmoid_derivative(z2)
    grad_y1 = w2.T @ grad_z2
    grad_z1 = grad_y1 * sigmoid_derivative(z1)
    return {
        "x": x, "w1": w1, "w2": w2, "w3": w3, "z1": z1, "y1": y1, "z2": z2, "y2": y2,
        "zO": zO, "yO": yO, "grad_zO": grad_zO, "grad_z2": grad_z2, "grad_z1": grad_z1,
    }


def test_mlp_forward_matches_manual_computation():
    demo = MultilayerNetworkDemo()
    m = _manual_mlp_forward_backward(demo)
    last = demo.checkpoint_frames()[-1].values
    np.testing.assert_allclose(last["z1"], m["z1"])
    np.testing.assert_allclose(last["y1"], m["y1"])
    np.testing.assert_allclose(last["z2"], m["z2"])
    np.testing.assert_allclose(last["y2"], m["y2"])
    assert last["zO"] == pytest.approx(m["zO"])
    assert last["yO"] == pytest.approx(m["yO"])
    assert 0.0 < last["yO"] < 1.0


def test_mlp_backward_gradients_match_the_chain_rule():
    demo = MultilayerNetworkDemo()
    m = _manual_mlp_forward_backward(demo)
    last = demo.checkpoint_frames()[-1].values
    assert last["grad_zO"] == pytest.approx(m["grad_zO"])
    np.testing.assert_allclose(last["grad_z2"], m["grad_z2"])
    np.testing.assert_allclose(last["grad_z1"], m["grad_z1"])


def test_mlp_walks_one_neuron_at_a_time_in_topological_order():
    demo = MultilayerNetworkDemo()
    checkpoints = [f for f in demo.checkpoint_frames() if f.values["active"]]
    order = [f.values["active"] for f in checkpoints]
    # forward left-to-right, then (after the loss checkpoint, still
    # spotlighting the output neuron) backward right-to-left.
    assert order == ["L1-A", "L1-B", "L2-C", "L2-D", "Saída", "Saída", "Saída", "L2-D", "L2-C", "L1-B", "L1-A"]


def test_mlp_sigmoid_inset_fields_match_the_active_neuron():
    # whichever neuron is "active" in a checkpoint, the spotlight fields
    # feeding the sigmoid inset must be *that* neuron's own z/y/slope --
    # not some other neuron's, and not stale from a previous step.
    demo = MultilayerNetworkDemo()
    m = _manual_mlp_forward_backward(demo)
    per_neuron_z = {
        "L1-A": m["z1"][0], "L1-B": m["z1"][1], "L2-C": m["z2"][0], "L2-D": m["z2"][1], "Saída": m["zO"],
    }
    for f in demo.checkpoint_frames():
        active = f.values["active"]
        if not active:
            continue
        assert f.values["active_z"] == pytest.approx(per_neuron_z[active])
        assert f.values["active_y"] == pytest.approx(sigmoid(per_neuron_z[active]))


def test_mlp_weights_actually_change_after_the_update_step():
    demo = MultilayerNetworkDemo()
    last = demo.checkpoint_frames()[-1].values
    assert not np.allclose(last["w1_updated"], last["w1"])
    assert not np.allclose(last["w2_updated"], last["w2"])
    assert not np.allclose(last["w3_updated"], last["w3"])


def test_mlp_reveals_never_go_backward():
    # once a neuron's forward/backward result is shown, it must stay shown
    # for every later checkpoint -- the network only ever grows on screen.
    demo = MultilayerNetworkDemo()
    reveal_keys = [
        "fwd_l1a", "fwd_l1b", "fwd_l2c", "fwd_l2d", "fwd_o", "loss_reveal",
        "bwd_o", "bwd_l2d", "bwd_l2c", "bwd_l1b", "bwd_l1a", "update_reveal",
    ]
    checkpoints = demo.checkpoint_frames()
    for key in reveal_keys:
        values = [float(f.values[key]) for f in checkpoints]
        assert all(a <= b + 1e-9 for a, b in zip(values, values[1:])), key
