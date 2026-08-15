import numpy as np
import pytest

from efficient_nn_lab.snn.encoding import direct_threshold_spikes
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif
from efficient_nn_lab.snn.surrogate import (
    fast_sigmoid_surrogate,
    heaviside,
    heaviside_derivative,
)
from efficient_nn_lab.snn.demos.lif_dynamics import LIFDynamicsDemo
from efficient_nn_lab.snn.demos.spike_generation import SpikeGenerationDemo
from efficient_nn_lab.snn.demos.surrogate_gradient import SurrogateGradientDemo


# -- LIF integration --------------------------------------------------------
def test_lif_no_current_never_spikes_and_decays_to_rest():
    current = constant_current(0.0, 30)
    trace = simulate_lif(current, LIFParams(v_rest=0.0))
    assert trace.spikes.sum() == 0
    assert trace.membrane[-1] == pytest.approx(0.0)


def test_lif_sufficient_current_spikes_and_resets():
    current = constant_current(0.30, 40, onset=0)
    params = LIFParams(tau=5.0, r=5.0, v_th=1.0, v_reset=0.0)
    trace = simulate_lif(current, params)
    assert trace.spikes.sum() >= 1
    spike_indices = np.nonzero(trace.spikes)[0]
    for idx in spike_indices:
        assert trace.membrane[idx] == pytest.approx(params.v_reset)


def test_lif_membrane_never_exceeds_threshold():
    current = constant_current(0.5, 50)
    params = LIFParams(tau=5.0, r=5.0, v_th=1.0)
    trace = simulate_lif(current, params)
    assert np.all(trace.membrane <= params.v_th + 1e-9)


# -- surrogate gradient ------------------------------------------------
def test_heaviside_is_a_true_step():
    x = np.array([-1.0, -0.001, 0.0, 0.001, 1.0])
    np.testing.assert_array_equal(heaviside(x), [0.0, 0.0, 1.0, 1.0, 1.0])


def test_heaviside_derivative_is_zero_everywhere_shown():
    x = np.linspace(-2, 2, 50)
    assert np.all(heaviside_derivative(x) == 0.0)


def test_surrogate_peaks_at_threshold_and_decays_away():
    peak = fast_sigmoid_surrogate(np.array([0.0]), k=5.0)[0]
    near = fast_sigmoid_surrogate(np.array([0.05]), k=5.0)[0]
    far = fast_sigmoid_surrogate(np.array([2.0]), k=5.0)[0]
    assert peak > near > far
    assert peak == pytest.approx(1.0)


# -- direct threshold spike encoding --------------------------------------
def test_direct_threshold_spikes_only_on_rising_edge():
    signal = np.array([0.0, 0.5, 0.5, 0.0, 0.5])
    spikes = direct_threshold_spikes(signal, level=0.4)
    np.testing.assert_array_equal(spikes, [0.0, 1.0, 0.0, 0.0, 1.0])


# -- demo modules -----------------------------------------------------------
def test_lif_dynamics_demo_has_a_spike_and_a_reset_phase():
    demo = LIFDynamicsDemo()
    phases = {f.values["phase"] for f in demo._frames}
    assert "spike + reset" in phases
    assert "repouso" in phases


def test_spike_generation_demo_reveals_progressively():
    demo = SpikeGenerationDemo()
    lengths = [len(f.values["signal"]) for f in demo._frames]
    assert lengths == sorted(lengths)
    assert lengths[-1] == 60


def test_lif_dynamics_only_marks_meaningful_moments_as_checkpoints():
    # every time-step is a frame (smooth sweep), but only phase changes
    # are checkpoints, so "Anterior"/"Proximo" moves meaningfully.
    demo = LIFDynamicsDemo()
    assert len(demo._frames) == 60
    assert demo.total_steps < 60
    assert demo.total_steps >= 3  # start, at least one spike, end


def test_surrogate_gradient_demo_morphs_true_derivative_into_surrogate():
    demo = SurrogateGradientDemo()
    checkpoints = demo.checkpoint_frames()
    labels = [f.label for f in checkpoints]
    assert labels == [
        "A função de disparo (forward)",
        "A derivada real",
        "O gradiente substituto",
        "Os dois juntos",
    ]
    true_deriv_curve = checkpoints[1].values["curve"]
    surrogate_curve = checkpoints[2].values["curve"]
    assert np.all(true_deriv_curve == 0.0)
    assert surrogate_curve.max() > 0.5
    # the animated transition between them actually interpolates the
    # array, peak growing from 0 toward the surrogate's peak.
    mid_tween = demo._frames[demo._checkpoint_frame_indices[1] + 3]
    assert 0.0 < mid_tween.values["curve"].max() < surrogate_curve.max()
