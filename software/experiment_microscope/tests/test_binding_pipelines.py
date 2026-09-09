"""nn_microscope.meeting01 + .thesis smoke + invariant checks (FIXME §49).

Skipped (with the build command) when the extension is absent.
"""

import numpy as np
import pytest

from experiment_microscope.paths import NN_BINDING_BUILD_HINT
from experiment_microscope.processing._binding import is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason=f"nn_microscope not built. Build: {NN_BINDING_BUILD_HINT}"
)


def test_meeting01_direct_encoding_is_identity():
    from experiment_microscope.processing import meeting01 as m

    w = np.random.default_rng(0).standard_normal((256, 1))
    np.testing.assert_allclose(m.encode(w, "direct", 0), w)


def test_meeting01_poisson_is_binary_and_seeded():
    from experiment_microscope.processing import meeting01 as m

    w = np.abs(np.random.default_rng(1).standard_normal((128, 1)))
    a = m.encode(w, "poisson", 7)
    b = m.encode(w, "poisson", 7)
    assert set(np.unique(a)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(a, b)  # deterministic for a fixed seed


def test_meeting01_dense_transform_is_passthrough():
    from experiment_microscope.processing import meeting01 as m

    w = np.random.default_rng(2).standard_normal((64, 1))
    np.testing.assert_allclose(m.architecture_transform(w, "dense", 0.9, 1.0), w)


def test_thesis_paraconsistent_formula_holds():
    from experiment_microscope.processing import paraconsistent as p

    nm_k = 0.5857864376269049
    rng = np.random.default_rng(3)
    fv, sid = [], []
    for subject in range(6):
        centre = rng.normal(subject * 4.0, 0.3, size=8)
        for _ in range(10):
            fv.append(centre + rng.normal(0, 0.1, size=8))
            sid.append(subject)
    s = p.score(fv, sid, "sep")
    assert np.isclose(s.g1, s.alpha - s.beta)
    assert np.isclose(s.g2, s.alpha + s.beta - 1.0)
    assert np.isclose(s.d_penalized, s.d_truth + nm_k * abs(s.g2))


def test_thesis_extract_handcrafted_is_finite():
    from experiment_microscope.processing import paraconsistent as p

    sig = np.sin(np.linspace(0, 40, 4096)) + 0.01 * np.random.default_rng(4).standard_normal(4096)
    feat = p.extract_handcrafted(sig, 1024.0, wavelet="daub4", dtwpt_level=4)
    assert len(feat) > 0
    assert np.all(np.isfinite(feat))
