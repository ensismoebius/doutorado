"""LiveVowelSnnDemo and LiveSpikeView, exercised with a SyntheticSource --
never a real microphone (there usually isn't one in CI, and a test that
depends on real ambient sound would not be repeatable anyway; see
live/audio_capture.py's SyntheticSource docstring).
"""

import numpy as np
import pytest

from efficient_nn_lab.live.audio_capture import SyntheticSource
from efficient_nn_lab.live.demo import LiveVowelSnnDemo
from efficient_nn_lab.live.lif_layer import LIFLayerParams
from efficient_nn_lab.live.vowel_model import VowelSnnWeights
from efficient_nn_lab.widgets.live_spike_view import LiveSpikeView

_SAMPLE_RATE = 16000
_N_BINS = 8


def _tiny_weights(seed: int = 0) -> VowelSnnWeights:
    rng = np.random.RandomState(seed)
    return VowelSnnWeights(
        w1=rng.normal(0.0, 0.3, size=(_N_BINS, 6)),
        w2=rng.normal(0.0, 0.3, size=(6, 5)),
        feature_min=np.zeros(_N_BINS),
        feature_max=np.ones(_N_BINS) * 5.0,
        n_bins=_N_BINS,
        sample_rate=_SAMPLE_RATE,
        window_seconds=0.1,
        t_steps=8,
        lif=LIFLayerParams(),
    )


def _tone(seconds: float, freq: float = 300.0) -> np.ndarray:
    t = np.arange(int(seconds * _SAMPLE_RATE)) / _SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_demo_without_weights_reports_no_weights(monkeypatch):
    # weights=None means "fall back to the trained weights file on disk if
    # one exists" (so the real app works out of the box once trained) --
    # force that fallback to miss too, so this test still covers the
    # "nothing trained yet" path even after a real vowel_snn_weights.npz
    # has been produced by live.train on this machine.
    monkeypatch.setattr("efficient_nn_lab.live.demo._try_load_default_weights", lambda: None)
    demo = LiveVowelSnnDemo(source=SyntheticSource(_tone(1.0)), weights=None)
    snapshot = demo.snapshot()
    assert snapshot["has_weights"] is False
    demo.start_capture()  # must be a safe no-op, not a crash, with no weights
    assert demo.is_capturing is False


def test_start_stop_capture_toggles_state():
    demo = LiveVowelSnnDemo(source=SyntheticSource(_tone(1.0)), weights=_tiny_weights())
    assert demo.is_capturing is False
    demo.start_capture()
    assert demo.is_capturing is True
    demo.stop_capture()
    assert demo.is_capturing is False


def test_poll_and_advance_updates_history_and_confidence():
    demo = LiveVowelSnnDemo(source=SyntheticSource(_tone(2.0), chunk_size=400), weights=_tiny_weights())
    demo.start_capture()
    for _ in range(20):
        demo.poll_and_advance()
    snapshot = demo.snapshot()
    assert snapshot["has_weights"] is True
    assert snapshot["predicted_class"] in snapshot["class_names"]
    assert any(len(row) for row in snapshot["hidden_spike_steps"] + snapshot["output_spike_steps"]), (
        "two full seconds of a loud tone through an untrained-but-random network "
        "produced literally zero spikes in either layer -- almost certainly a wiring bug"
    )


def test_poll_and_advance_is_noop_when_not_capturing():
    demo = LiveVowelSnnDemo(source=SyntheticSource(_tone(1.0)), weights=_tiny_weights())
    demo.poll_and_advance()  # never started
    snapshot = demo.snapshot()
    assert not any(len(row) for row in snapshot["hidden_spike_steps"])


@pytest.mark.usefixtures("qapp")
def test_live_spike_view_renders_without_weights():
    view = LiveSpikeView()
    demo = LiveVowelSnnDemo(source=SyntheticSource(_tone(1.0)), weights=None)
    view.render(demo.snapshot())  # must not raise


@pytest.mark.usefixtures("qapp")
def test_live_spike_view_renders_a_live_snapshot():
    view = LiveSpikeView()
    demo = LiveVowelSnnDemo(source=SyntheticSource(_tone(2.0), chunk_size=400), weights=_tiny_weights())
    demo.start_capture()
    for _ in range(20):
        demo.poll_and_advance()
    view.render(demo.snapshot())  # must not raise
