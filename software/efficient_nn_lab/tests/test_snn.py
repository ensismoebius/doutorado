import itertools

import numpy as np
import pytest

from efficient_nn_lab.snn.encoding import (
    first_spike_time,
    latency_quantization_error,
    latency_rms_error,
    latency_spike_time,
    load_grayscale_image,
    poisson_noise_sigma,
    poisson_rms_error,
    poisson_spike_frames,
    poisson_spikes,
    spike_probability,
    spike_time_grad_is_live,
    threshold_crossing_spikes,
)
from efficient_nn_lab.snn.lif import LIFParams, constant_current, simulate_lif
from efficient_nn_lab.snn.normalization import fit_zscore, leaky_fit_zscore, zscore
from efficient_nn_lab.snn.rate_reg import rate_reg_loss, rate_reg_push
from efficient_nn_lab.snn.surrogate import (
    fast_sigmoid,
    fast_sigmoid_surrogate,
    heaviside,
    heaviside_derivative,
)
from efficient_nn_lab.snn.tdbn import tdbn_transform
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
    assert peak == pytest.approx(2.5)  # k/2


def test_surrogate_has_unit_area_like_the_step_it_replaces():
    # The step jumps by 1, so a stand-in for its derivative should carry
    # area 1 for every k -- otherwise k silently rescales the gradient.
    x = np.linspace(-400.0, 400.0, 2_000_001)
    for k in (1.0, 5.0, 10.0):
        area = np.trapezoid(fast_sigmoid_surrogate(x, k=k), x)
        assert area == pytest.approx(1.0, abs=5e-3)


def test_fast_sigmoid_is_a_smooth_step_that_sharpens_with_k():
    # It must span the step's own range (0..1) and get CLOSER to the step
    # as k grows -- the earlier 0.5 + x/(1+k|x|) only spanned 0.5 +- 1/k,
    # flattening as k grew, the opposite of what the slider teaches.
    x = np.linspace(-2.0, 2.0, 401)
    step = heaviside(x)
    errors = []
    for k in (1.0, 5.0, 10.0):
        s = fast_sigmoid(x, k=k)
        assert np.all((s > 0.0) & (s < 1.0))
        errors.append(float(np.mean(np.abs(s - step))))
    assert errors[0] > errors[1] > errors[2]
    assert fast_sigmoid(np.array([-1e6]), k=5.0)[0] == pytest.approx(0.0, abs=1e-6)
    assert fast_sigmoid(np.array([1e6]), k=5.0)[0] == pytest.approx(1.0, abs=1e-6)


def test_fast_sigmoid_is_the_true_antiderivative_of_the_surrogate():
    # fast_sigmoid must not just *look* like an S-curve near the surrogate
    # gradient -- its numerical slope has to match fast_sigmoid_surrogate
    # at every point, or the two plotted curves would be lying about being
    # a function/derivative pair.
    x = np.linspace(-2.0, 2.0, 400)
    sigmoid = fast_sigmoid(x, k=5.0)
    surrogate = fast_sigmoid_surrogate(x, k=5.0)
    numeric_slope = np.gradient(sigmoid, x)
    # relative: the peak is k/2 = 2.5, and the finite-difference error is
    # largest right at that cusp (~1% there, far less elsewhere).
    np.testing.assert_allclose(numeric_slope, surrogate, rtol=2e-2, atol=1e-9)
    zero_idx = int(np.abs(x).argmin())
    assert sigmoid[0] < sigmoid[zero_idx] < sigmoid[-1]
    # exactly 0.5 AT the threshold (the 400-point grid has no x = 0 sample)
    assert fast_sigmoid(np.array([0.0]), k=5.0)[0] == pytest.approx(0.5)


# -- direct threshold spike encoding --------------------------------------
def test_threshold_crossing_spikes_only_on_rising_edge():
    signal = np.array([0.0, 0.5, 0.5, 0.0, 0.5])
    spikes = threshold_crossing_spikes(signal, level=0.4)
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


def test_load_grayscale_image_has_real_variance():
    # guards against the image loading "successfully" but as a flat/
    # degenerate array (wrong channel math, corrupted asset) -- a correct
    # photo must contain real contrast, not just the right shape/range.
    image = load_grayscale_image(_IMAGE_PATH, size=(108, 192))
    assert image.shape == (108, 192)
    assert image.std() > 0.05
    assert image.max() - image.min() > 0.5


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


# -- tdBN: verified against software/nn's own worked numeric example
# (.wiki/Concepts/Threshold-Dependent-Batch-Normalization.md) --------------
def test_tdbn_transform_matches_wiki_worked_example():
    x = np.array([0.0, 2.0, 4.0, 6.0])
    y_vth1 = tdbn_transform(x, v_th=1.0)
    np.testing.assert_allclose(y_vth1, [-1.3416, -0.4472, 0.4472, 1.3416], atol=1e-4)
    y_vth2 = tdbn_transform(x, v_th=2.0)
    np.testing.assert_allclose(y_vth2, [-2.6833, -0.8944, 0.8944, 2.6833], atol=1e-4)


# -- firing-rate regularization (.wiki/Concepts/Spike-Rate-Regularization.md)
def test_rate_reg_pushes_below_floor_rate_upward():
    push = rate_reg_push(mean_rate=0.02, lambda_reg=0.5, r_min=0.05)
    assert push > 0.0


def test_rate_reg_no_push_inside_band():
    assert rate_reg_push(mean_rate=0.3, lambda_reg=0.5, r_min=0.05, r_max=0.8) == pytest.approx(0.0)
    assert rate_reg_loss(mean_rate=0.3, lambda_reg=0.5, r_min=0.05, r_max=0.8) == pytest.approx(0.0)


def test_rate_reg_pushes_above_ceiling_downward():
    push = rate_reg_push(mean_rate=0.95, lambda_reg=0.5, r_max=0.8)
    assert push < 0.0


# -- encoding noise floor / no-spike guard (.wiki/Concepts/Spike-Encoding.md)
def test_poisson_noise_sigma_worst_case_at_t16():
    assert poisson_noise_sigma(16) == pytest.approx(0.125, abs=1e-9)


def test_latency_quantization_error_is_half_the_level_spacing():
    # T frames 0..T-1 hold T levels spaced 1/(T-1) apart (round((1-x)(T-1)),
    # the formula of both software/nn encoders) -> worst error 0.5/(T-1).
    assert latency_quantization_error(16) == pytest.approx(0.5 / 15, abs=1e-12)
    assert latency_quantization_error(16) == pytest.approx(0.0333, abs=1e-4)
    with pytest.raises(ValueError):
        latency_quantization_error(1)


def test_latency_spike_time_matches_the_software_nn_encoders():
    # t = round((1 - x) * (T - 1)), frames 0..T-1 (Meeting01Encoding.cpp,
    # ThesisFeatureExtractionInternal.hpp); every value fires exactly once.
    assert [latency_spike_time(x, time_steps=16) for x in (1.0, 0.8, 0.4, 0.2, 0.0)] == [0, 3, 9, 12, 15]
    # half away from zero, like std::llround -- Python's round(7.5) is 8 too,
    # but round(0.5) would be 0, so T=2 at x=0.5 pins the rounding rule.
    assert latency_spike_time(0.5, time_steps=2) == 1


def test_first_spike_time_never_fires_defaults_to_t():
    assert first_spike_time(np.zeros(16), time_steps=16) == 16
    assert first_spike_time(np.array([0, 0, 1, 0]), time_steps=4) == 2


def test_spike_time_grad_is_live_false_exactly_at_no_spike_deadlock():
    assert spike_time_grad_is_live(predicted_time=5, time_steps=16) is True
    assert spike_time_grad_is_live(predicted_time=16, time_steps=16) is False


# -- normalization: per-feature (fit-once) vs per-window, and the leaky-fit
# hazard (thesis chapter 07, sec:normalizacaoEntrada) ----------------------
def test_zscore_worked_examples():
    """The thesis's own numbers (chapter 07): the audio training column
    [2, 4, 4, 6] scores a test x = 5 as 0.7071; the EEG window
    [10, 12, 14, 16] uV scores its first sample as -1.3416."""
    mean, std = fit_zscore(np.array([2.0, 4.0, 4.0, 6.0]))
    assert (mean, std**2) == pytest.approx((4.0, 2.0))
    assert zscore(5.0, mean, std) == pytest.approx(0.7071, abs=1e-4)
    mean, std = fit_zscore(np.array([10.0, 12.0, 14.0, 16.0]))
    assert (mean, std**2) == pytest.approx((13.0, 5.0))
    assert zscore(10.0, mean, std) == pytest.approx(-1.3416, abs=1e-4)


def test_leaky_fit_differs_from_train_only_fit():
    train = np.array([1.0, 2.0, 3.0])
    test = np.array([100.0, 101.0])
    train_only = fit_zscore(train)
    leaky = leaky_fit_zscore(train, test)
    assert train_only != leaky


def test_rate_reg_demo_mixed_layer_mean_hides_both_sick_units():
    # software/nn penalizes ONE mean per layer (spikes.sum() / n): a layer
    # holding a dead unit and a bursting unit averages into the band, so
    # neither unit is pushed. The demo must show that blind spot.
    from efficient_nn_lab.snn.demos.firing_rate_reg import FiringRateRegDemo

    demo = FiringRateRegDemo()
    mixed = next(f.values for f in demo.checkpoint_frames() if f.values["mixed_reveal"] >= 1.0)
    assert mixed["mixed_dead"] < mixed["r_min"]
    assert mixed["mixed_burst"] > mixed["r_max"]
    assert mixed["r_min"] <= mixed["mixed_mean"] <= mixed["r_max"]
    assert mixed["mixed_push"] == pytest.approx(0.0)
    # while each uniformly sick layer IS pushed back toward the band
    first = demo.checkpoint_frames()[0].values
    assert first["push_dead"] > 0.0 > first["push_burst"]


def test_lif_dynamics_demo_shows_the_leak_decaying_after_the_current_switches_off():
    # With a constant current V only rises, so the "vazamento" phase used to
    # be unreachable. The pulse input makes the leak visible on its own.
    demo = LIFDynamicsDemo()
    leak = [f for f in demo._frames if f.values["phase"] == "vazamento"]
    assert leak
    membrane = demo._frames[-1].values["membrane"]
    first_leak = len(demo._frames) - len(leak)
    decay = membrane[first_leak:]
    assert all(a > b for a, b in itertools.pairwise(decay))
    assert any(f.is_checkpoint for f in leak)


def test_lif_dynamics_demo_explains_subthreshold_equilibrium():
    demo = LIFDynamicsDemo()
    demo.set_parameter("amplitude", 0.1)  # R*I = 0.5 < V_th = 1
    # on a CHECKPOINT: the window sizes the explanation label from
    # checkpoint texts only, so a text shown only between checkpoints is
    # never measured and can clip
    texts = {f.explanation for f in demo.checkpoint_frames()}
    assert any("nunca" in t and "dispara" in t for t in texts)
    assert not any(f.values["phase"] == "spike + reset" for f in demo._frames)


def test_noise_floor_rms_formulas_match_monte_carlo():
    # Both floors on the encoding-noise plot must be the SAME statistic --
    # RMS error with x uniform on [0, 1] -- or the shared axis lies.
    rng = np.random.default_rng(0)
    t_steps = 16
    x = rng.uniform(0.0, 1.0, 200_000)
    counts = rng.binomial(t_steps, x)
    poisson_rms = float(np.sqrt(np.mean((counts / t_steps - x) ** 2)))
    assert poisson_rms == pytest.approx(poisson_rms_error(t_steps), rel=0.01)
    frames = np.array([latency_spike_time(v, t_steps) for v in x[:20_000]])
    decoded = 1.0 - frames / (t_steps - 1)
    latency_rms = float(np.sqrt(np.mean((decoded - x[:20_000]) ** 2)))
    assert latency_rms == pytest.approx(latency_rms_error(t_steps), rel=0.02)
