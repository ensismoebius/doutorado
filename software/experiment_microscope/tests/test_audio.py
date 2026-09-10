import numpy as np
import pytest

pytest.importorskip("PySide6")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.data.adapters import Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.viz import audio


def test_player_rejects_non_audio_and_bad_samples(qapp):
    p = audio.AudioPlayer()
    with pytest.raises(RuntimeError, match="not audible|EEG"):
        p.play(np.zeros(64), sample_rate=1024.0)   # EEG rate
    with pytest.raises(RuntimeError, match="NaN"):
        p.play(np.array([0.0, np.nan, 1.0]), sample_rate=16000.0)
    with pytest.raises(RuntimeError, match="empty"):
        p.play(np.array([]), sample_rate=16000.0)


class _FakeAdapter:
    def __init__(self, sig):
        self._sig = sig

    def load_signal(self, node):
        return self._sig


def _view_with(sig):
    from experiment_microscope.views.signal_view import SignalView

    repo = DataRepository()
    repo._adapters["thesis"] = _FakeAdapter(sig)
    v = SignalView(repo)
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    return v


def test_listen_button_enabled_only_for_audio(qapp):
    voice = Signal1D(samples=np.sin(np.linspace(0, 40, 8000)), sample_rate=8000.0,
                     origin=Origin.MEASURED, label="voice")
    assert _view_with(voice)._listen.isEnabled() == audio.is_available()

    eeg = Signal1D(samples=np.zeros((6, 512)), sample_rate=1024.0,
                   origin=Origin.MEASURED, label="eeg")
    assert not _view_with(eeg)._listen.isEnabled()
