import numpy as np
import pytest

from efficient_nn_lab.snn.encoding import (
    direct_threshold_spikes,
    load_grayscale_image,
    poisson_spike_frames,
    poisson_spikes,
    spike_probability,
)
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif
from efficient_nn_lab.snn.surrogate import (
    fast_sigmoid,
    fast_sigmoid_surrogate,
    heaviside,
    heaviside_derivative,
)
from efficient_nn_lab.snn.demos.lif_dynamics import LIFDynamicsDemo
from efficient_nn_lab.snn.demos.poisson_image_coding import _IMAGE_PATH, PoissonImageCodingDemo
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


def test_fast_sigmoid_is_the_true_antiderivative_of_the_surrogate():
    # fast_sigmoid must not just *look* like an S-curve near the surrogate
    # gradient -- its numerical slope has to match fast_sigmoid_surrogate
    # at every point, or the two plotted curves would be lying about being
    # a function/derivative pair.
    x = np.linspace(-2.0, 2.0, 400)
    sigmoid = fast_sigmoid(x, k=5.0)
    surrogate = fast_sigmoid_surrogate(x, k=5.0)
    numeric_slope = np.gradient(sigmoid, x)
    np.testing.assert_allclose(numeric_slope, surrogate, atol=1.5e-2)
    zero_idx = int(np.abs(x).argmin())
    assert sigmoid[0] < sigmoid[zero_idx] < sigmoid[-1]
    assert sigmoid[zero_idx] == pytest.approx(0.5, abs=1e-2)


# -- direct threshold spike encoding --------------------------------------
def test_direct_threshold_spikes_only_on_rising_edge():
    signal = np.array([0.0, 0.5, 0.5, 0.0, 0.5])
    spikes = direct_threshold_spikes(signal, level=0.4)
    np.testing.assert_array_equal(spikes, [0.0, 1.0, 0.0, 0.0, 1.0])


# -- Poisson / rate coding --------------------------------------------------
def test_spike_probability_scales_with_intensity_and_clips_negatives():
    signal = np.array([-1.0, 0.0, 0.5, 1.0])
    prob = spike_probability(signal, max_rate=0.8)
    np.testing.assert_allclose(prob, [0.0, 0.0, 0.4, 0.8])


def test_poisson_spikes_is_deterministic_given_the_same_seed():
    signal = np.linspace(0.0, 1.0, 50)
    first = poisson_spikes(signal, max_rate=0.9, seed=42)
    second = poisson_spikes(signal, max_rate=0.9, seed=42)
    np.testing.assert_array_equal(first, second)


def test_poisson_spikes_fires_more_often_where_intensity_is_higher():
    # not a per-sample guarantee (it's a draw), but over many repeated
    # signals at a fixed intensity the empirical rate should track the
    # requested probability -- the defining property of rate coding.
    n = 4000
    low_signal = np.full(n, 0.1)
    high_signal = np.full(n, 0.9)
    low_rate = poisson_spikes(low_signal, max_rate=0.9, seed=7).mean()
    high_rate = poisson_spikes(high_signal, max_rate=0.9, seed=7).mean()
    assert low_rate < high_rate
    assert high_rate == pytest.approx(0.9 * 0.9, abs=0.03)


# -- Poisson coding on an image ----------------------------------------------
def test_load_grayscale_image_is_normalized_and_resized():
    image = load_grayscale_image(_IMAGE_PATH, size=(16, 20))
    assert image.shape == (16, 20)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_poisson_spike_frames_shape_and_determinism():
    intensity = np.array([[0.0, 1.0], [0.5, 0.2]])
    frames = poisson_spike_frames(intensity, n_steps=30, max_rate=0.9, seed=42)
    assert frames.shape == (30, 2, 2)
    assert np.all((frames == 0.0) | (frames == 1.0))
    # a fully dark pixel never spikes; a fully bright one spikes often but
    # not on literally every single step, at a moderate max_rate.
    assert frames[:, 0, 0].sum() == 0
    assert 0 < frames[:, 0, 1].sum() < 30
    again = poisson_spike_frames(intensity, n_steps=30, max_rate=0.9, seed=42)
    np.testing.assert_array_equal(frames, again)


def test_poisson_image_coding_demo_has_30_steps_from_the_real_image():
    demo = PoissonImageCodingDemo()
    checkpoints = demo.checkpoint_frames()
    assert len(checkpoints) == 30
    assert checkpoints[0].values["t"] == 0
    assert checkpoints[-1].values["t"] == 29
    image = checkpoints[0].values["image"]
    assert image.shape == (108, 192)
    for cp in checkpoints:
        frame = cp.values["frame"]
        assert frame.shape == image.shape
        assert np.all((frame == 0.0) | (frame == 1.0))


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


def test_surrogate_gradient_demo_sweeps_gradient_and_sigmoid_together():
    demo = SurrogateGradientDemo()
    checkpoints = demo.checkpoint_frames()
    labels = [f.label for f in checkpoints]
    assert labels == [
        "A função de disparo (forward)",
        "A sigmoide suave por trás do gradiente substituto",
        "A derivada real",
        "O gradiente substituto",
        "Os dois juntos",
    ]
    true_derivative = checkpoints[2].values["true_derivative"]
    surrogate = checkpoints[3].values["surrogate"]
    assert np.all(true_derivative == 0.0)
    assert surrogate.max() > 0.5
    assert checkpoints[2].values["draw_reveal"] == pytest.approx(0.0)
    assert checkpoints[3].values["draw_reveal"] == pytest.approx(1.0)
    # the sweep genuinely progresses across many interior frames -- a
    # partially-drawn state exists between "A derivada real" (nothing
    # drawn) and "O gradiente substituto" (fully drawn), not an instant
    # jump from one to the other.
    mid_tween = demo._frames[demo._checkpoint_frame_indices[2] + 20]
    assert 0.0 < mid_tween.values["draw_reveal"] < 1.0
    # the underlying curves themselves are constant across the sweep --
    # only how much of them is drawn changes -- so their values never
    # morph, unlike the old height-tweening design this replaces.
    np.testing.assert_array_equal(mid_tween.values["sigmoid"], checkpoints[3].values["sigmoid"])
    np.testing.assert_array_equal(mid_tween.values["surrogate"], checkpoints[3].values["surrogate"])
