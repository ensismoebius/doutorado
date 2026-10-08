"""The live-mic vowel-classification demo itself.

Every other `DemoModule` in this app satisfies `core/demo.py`'s contract:
"everything... precomputed once, deterministically, in initialize()". A
live microphone cannot be precomputed -- it hasn't happened yet, and it
is never the same twice. Rather than bending that contract (or every
caller that trusts it) to fit one demo, this class keeps it technically:
`_build_frames` returns exactly one static placeholder `Frame`, so
`DemoModule.initialize()`'s checks all pass trivially, and the SIDEBAR
and FRAME-CAPTION machinery treat it like any other demo. The actual live
behavior lives entirely in `start_capture`/`stop_capture`/`poll_and_advance`/
`snapshot`, which `main_window.py` calls directly instead of going through
`StepPlayer` (see `DemoModule.supports_live_capture` and `_select_demo`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.live.audio_capture import AudioSource, MicrophoneSource
from efficient_nn_lab.live.features import log_energy_bins
from efficient_nn_lab.live.vowel_model import VOWELS, VowelSnnWeights, forward, load_weights

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "vowel_snn_weights.npz"
#: How many recent LIF time-steps the rolling raster keeps on screen.
#: 10 processed windows' worth -- enough to read as a moving raster, not
#: so much that the oldest spikes on screen are from a minute ago.
_HISTORY_STEPS_WINDOWS = 10
#: How fast the live confidence readout follows a new window's firing
#: rate. Low enough that one noisy window (coughing, a chair creak)
#: doesn't yank the prediction around; high enough to visibly react within
#: a handful of windows (~1s) once someone actually speaks.
_CONFIDENCE_EMA = 0.35


def _try_load_default_weights() -> VowelSnnWeights | None:
    if not _DEFAULT_WEIGHTS_PATH.exists():
        return None
    return load_weights(str(_DEFAULT_WEIGHTS_PATH))


class LiveVowelSnnDemo(DemoModule):
    title = "SNN -> Classificação de vogais ao vivo (microfone)"
    slug = "live.vowel_snn"
    supports_live_capture = True
    description = (
        "O único demo deste app com um classificador REALMENTE treinado -- "
        "todos os outros usam dados fixos ou sintéticos, de propósito (ver "
        "Comparação -> Autoencoders). Uma SNN (Spiking Neural Network, rede "
        "neural de pulso) pequena, com uma camada oculta e uma de saída, "
        "recebe o microfone ao vivo, codifica por Poisson a energia "
        "espectral do som e tenta reconhecer qual vogal (a/e/i/o/u) está "
        "sendo falada -- os disparos das duas camadas e a confiança por "
        "vogal aparecem em tempo real."
    )

    def __init__(self, source: AudioSource | None = None, weights: VowelSnnWeights | None = None) -> None:
        self._source: AudioSource = source if source is not None else MicrophoneSource()
        self._weights = weights if weights is not None else _try_load_default_weights()
        self.is_capturing = False
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._confidence = np.full(len(VOWELS), 1.0 / len(VOWELS))
        if self._weights is not None:
            n_hidden = self._weights.w1.shape[1]
            n_out = self._weights.w2.shape[1]
            history = _HISTORY_STEPS_WINDOWS * self._weights.t_steps
            self._hidden_history = np.zeros((n_hidden, history), dtype=bool)
            self._output_history = np.zeros((n_out, history), dtype=bool)
        else:
            self._hidden_history = np.zeros((0, 0), dtype=bool)
            self._output_history = np.zeros((0, 0), dtype=bool)
        super().__init__()

    # -- DemoModule contract (see module docstring) ----------------------
    def _build_frames(self) -> list[Frame]:
        return [Frame("Captura ao vivo", {"kind": "live_vowel_snn"}, self.description)]

    # -- live capture, called directly by main_window.py -----------------
    def start_capture(self) -> None:
        if self.is_capturing or self._weights is None:
            return
        self._source.start()
        self.is_capturing = True

    def stop_capture(self) -> None:
        if not self.is_capturing:
            return
        self._source.stop()
        self.is_capturing = False

    def poll_and_advance(self) -> None:
        """Called once per UI tick while capturing: drain new audio, run
        inference on every full window it completes, update the rolling
        raster history and the smoothed confidence."""
        if not self.is_capturing or self._weights is None:
            return
        new_samples = self._source.read_available()
        if new_samples.size:
            self._audio_buffer = np.concatenate([self._audio_buffer, new_samples])
        window = int(self._weights.window_seconds * self._weights.sample_rate)
        hop = max(1, window // 4)
        while self._audio_buffer.size >= window:
            self._step(self._audio_buffer[:window])
            self._audio_buffer = self._audio_buffer[hop:]

    def _step(self, window_samples: np.ndarray) -> None:
        weights = self._weights
        assert weights is not None
        features = log_energy_bins(window_samples, weights.sample_rate, weights.n_bins, weights.fmin, weights.fmax)
        hidden_cache, output_cache = forward(features, weights, seed=None)

        t_steps = weights.t_steps
        self._hidden_history = np.concatenate([self._hidden_history[:, t_steps:], hidden_cache.spikes.T.astype(bool)], axis=1)
        self._output_history = np.concatenate([self._output_history[:, t_steps:], output_cache.spikes.T.astype(bool)], axis=1)

        rate = output_cache.spikes.mean(axis=0)
        self._confidence = (1.0 - _CONFIDENCE_EMA) * self._confidence + _CONFIDENCE_EMA * rate

    # -- what the view needs ---------------------------------------------
    def snapshot(self) -> dict[str, object]:
        if self._weights is None:
            return {
                "has_weights": False,
                "is_capturing": False,
                "hidden_spike_steps": [],
                "output_spike_steps": [],
                "history_steps": 0,
                "class_names": VOWELS,
                "class_confidences": [0.0] * len(VOWELS),
                "predicted_class": None,
            }
        history_steps = self._hidden_history.shape[1]
        return {
            "has_weights": True,
            "is_capturing": self.is_capturing,
            "hidden_spike_steps": [np.nonzero(row)[0].tolist() for row in self._hidden_history],
            "output_spike_steps": [np.nonzero(row)[0].tolist() for row in self._output_history],
            "history_steps": history_steps,
            "class_names": self._weights.class_names,
            "class_confidences": self._confidence.tolist(),
            "predicted_class": self._weights.class_names[int(np.argmax(self._confidence))],
        }
