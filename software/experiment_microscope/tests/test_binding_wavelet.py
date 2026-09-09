"""nn_microscope.wavelet parity + shape checks (FIXME §49).

Skipped when the extension is not built — with the build command in the skip
reason, per the no-fallback policy.
"""

import numpy as np
import pytest

from experiment_microscope.paths import NN_BINDING_BUILD_HINT
from experiment_microscope.processing._binding import BindingUnavailableError

try:
    from experiment_microscope.processing import wavelet as wl

    _AVAILABLE = True
except Exception:  # noqa: BLE001
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _AVAILABLE, reason=f"nn_microscope not built. Build: {NN_BINDING_BUILD_HINT}"
)


def _binding_ok() -> bool:
    try:
        wl.load_binding()
        return True
    except BindingUnavailableError:
        return False


def test_packet_leaf_count_is_two_to_the_level():
    if not _binding_ok():
        pytest.skip(f"nn_microscope .so missing. Build: {NN_BINDING_BUILD_HINT}")
    sig = np.sin(np.linspace(0, 20, 256))
    for level in (2, 3, 4):
        d = wl.decompose(sig, "haar", "packet", level)
        assert d.leaf_count == 2**level
        assert d.subband_energies.size == 2**level


def test_decompose_is_deterministic():
    if not _binding_ok():
        pytest.skip("nn_microscope .so missing")
    sig = np.cos(np.linspace(0, 8, 128)) + 0.1
    a = wl.decompose(sig, "daub4", "regular", 3).transformed_signal
    b = wl.decompose(sig, "daub4", "regular", 3).transformed_signal
    np.testing.assert_array_equal(a, b)
    assert a.size == 128


def test_unknown_wavelet_raises_with_remedy():
    if not _binding_ok():
        pytest.skip("nn_microscope .so missing")
    with pytest.raises(ValueError) as exc:
        wl.decompose([0.0] * 64, "morlet", "packet", 2)
    assert "daub" in str(exc.value)
