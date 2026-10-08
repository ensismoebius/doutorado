import numpy as np
import pytest

from efficient_nn_lab.live.lif_layer import LIFLayerParams, backward_layer, forward_layer
from efficient_nn_lab.snn.surrogate import fast_sigmoid


def test_forward_layer_silent_with_no_input_current():
    input_spikes = np.zeros((10, 3))
    weights = np.ones((3, 2))
    cache = forward_layer(input_spikes, weights, LIFLayerParams())
    assert cache.spikes.sum() == 0


def test_forward_layer_fires_once_weight_pushes_past_threshold():
    # A single input spike at t=0, weight big enough to clear v_th alone.
    input_spikes = np.zeros((5, 1))
    input_spikes[0, 0] = 1.0
    weights = np.array([[2.0]])
    cache = forward_layer(input_spikes, weights, LIFLayerParams(v_th=1.0))
    assert cache.spikes[0, 0] == 1.0


def test_forward_layer_resets_after_firing():
    # Keep feeding the same current once it fires: with the recurrence's
    # (1 - s_prev) reset, membrane after a spike restarts near the new
    # input alone, not the pre-spike value plus the new input.
    input_spikes = np.ones((3, 1))
    weights = np.array([[2.0]])
    params = LIFLayerParams(v_th=1.0, tau=5.0)
    cache = forward_layer(input_spikes, weights, params)
    assert cache.spikes[0, 0] == 1.0
    # Immediately after the reset, membrane == this step's own current
    # (2.0), not decay*2.0 + 2.0 -- the reset actually zeroed it first.
    assert cache.membrane[1, 0] == pytest.approx(2.0)


def _smooth_forward_with_fixed_reset(
    input_spikes: np.ndarray, weights: np.ndarray, params: LIFLayerParams, reset_mask: np.ndarray
) -> np.ndarray:
    """Exactly `forward_layer`'s membrane recurrence, but with the spike
    nonlinearity replaced by the smooth `fast_sigmoid` (so this whole
    function is differentiable) and the `(1 - s_prev)` reset held FIXED at
    `reset_mask` rather than recomputed from the (now smooth) spikes.

    This fixed-reset choice is deliberate, not a simplification of
    convenience: it is exactly the dependency `backward_layer` computes a
    gradient FOR (see its module docstring -- the reset gate's dependence
    on the previous spike is treated as constant, not differentiated
    through). Checking against the full true gradient of a smooth forward
    pass would legitimately disagree with `backward_layer`'s answer, by
    design; checking against THIS -- the smooth forward pass restricted to
    the same dependency `backward_layer` actually differentiates -- is what
    verifies `backward_layer`'s formula is correct for what it claims to
    compute.
    """
    t_steps = input_spikes.shape[0]
    n_out = weights.shape[1]
    v = np.zeros(n_out)
    spikes = np.zeros((t_steps, n_out))
    for t in range(t_steps):
        current = input_spikes[t] @ weights
        v = params.decay * v * reset_mask[t] + current
        spikes[t] = fast_sigmoid(v - params.v_th, k=params.surrogate_k)
    return spikes


def _random_problem(seed: int, t_steps=6, n_in=3, n_out=2):
    rng = np.random.RandomState(seed)
    input_spikes = (rng.random_sample((t_steps, n_in)) < 0.5).astype(float)
    weights = rng.normal(0.0, 0.7, size=(n_in, n_out))
    return input_spikes, weights


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_backward_layer_weight_gradient_matches_finite_differences(seed):
    input_spikes, weights = _random_problem(seed)
    params = LIFLayerParams(v_th=0.5, tau=4.0, surrogate_k=5.0)

    cache = forward_layer(input_spikes, weights, params)
    n_out = weights.shape[1]
    s_prev = np.vstack([np.zeros((1, n_out)), cache.spikes[:-1]])
    reset_mask = 1.0 - s_prev  # fixed at the base point -- see helper's docstring

    def loss_fn(w: np.ndarray) -> float:
        smooth_spikes = _smooth_forward_with_fixed_reset(input_spikes, w, params, reset_mask)
        return float(np.sum(smooth_spikes**2))

    smooth_spikes_at_base = _smooth_forward_with_fixed_reset(input_spikes, weights, params, reset_mask)
    grad_spikes = 2.0 * smooth_spikes_at_base
    analytic_grad_w, _ = backward_layer(cache, weights, params, grad_spikes)

    eps = 1e-5
    numeric_grad_w = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            w_plus, w_minus = weights.copy(), weights.copy()
            w_plus[i, j] += eps
            w_minus[i, j] -= eps
            numeric_grad_w[i, j] = (loss_fn(w_plus) - loss_fn(w_minus)) / (2 * eps)

    np.testing.assert_allclose(analytic_grad_w, numeric_grad_w, atol=1e-4, rtol=1e-3)


def test_backward_layer_output_shapes():
    input_spikes, weights = _random_problem(seed=3, n_in=4, n_out=5)
    params = LIFLayerParams()
    cache = forward_layer(input_spikes, weights, params)
    grad_spikes = np.ones_like(cache.spikes)
    grad_w, grad_input = backward_layer(cache, weights, params, grad_spikes)
    assert grad_w.shape == weights.shape
    assert grad_input.shape == input_spikes.shape
