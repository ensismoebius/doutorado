import numpy as np
import pytest

from efficient_nn_lab.comparison.autoencoder_synthetic import (
    TRIVIAL_MSE_BY_TARGET,
    WINDOW,
    fit_pca,
    mse,
    pca_reconstruct,
    synthetic_window,
    window_set,
    zscore_window,
)


def test_windows_are_z_scored_like_the_meeting01_loaders():
    window = synthetic_window(np.random.RandomState(0))
    assert window.shape == (WINDOW,)
    assert window.mean() == pytest.approx(0.0, abs=1e-12)
    assert window.std() == pytest.approx(1.0, abs=1e-12)


def test_window_set_is_deterministic():
    np.testing.assert_array_equal(window_set(5), window_set(5))


def test_mean_frame_reference_scores_about_one_on_z_scored_windows():
    """Predicting the training mean (about 0 everywhere) on a window of
    variance 1 costs MSE about 1 -- the "learned nothing" anchor the demo uses."""
    train = window_set(400)
    test = synthetic_window(np.random.RandomState(7))
    assert mse(test, fit_pca(train).mean) == pytest.approx(1.0, abs=0.1)


def test_pca_error_never_grows_with_more_components_on_training_windows():
    train = window_set(200)
    pca = fit_pca(train)
    errors = [np.mean([mse(w, pca_reconstruct(pca, w, k)) for w in train]) for k in (1, 4, 16, 64)]
    assert all(a >= b - 1e-12 for a, b in zip(errors, errors[1:]))


def test_pca_with_every_component_reconstructs_exactly():
    train = window_set(300)
    pca = fit_pca(train)
    window = train[3]
    assert mse(window, pca_reconstruct(pca, window, len(pca.components))) == pytest.approx(0.0, abs=1e-20)


def test_pca_rejects_an_impossible_k():
    pca = fit_pca(window_set(20))
    with pytest.raises(ValueError):
        pca_reconstruct(pca, window_set(1)[0], 0)


def test_trivial_mse_is_the_target_variance():
    """B2's mechanism: a mean predictor's MSE IS the target's variance, so a
    sparser target scores lower without any skill."""
    window = zscore_window(np.random.RandomState(1).normal(size=WINDOW))
    assert mse(window, np.full(WINDOW, window.mean())) == pytest.approx(window.var(), abs=1e-12)
    assert TRIVIAL_MSE_BY_TARGET["direta"] > TRIVIAL_MSE_BY_TARGET["Poisson"] > TRIVIAL_MSE_BY_TARGET["latência"]
