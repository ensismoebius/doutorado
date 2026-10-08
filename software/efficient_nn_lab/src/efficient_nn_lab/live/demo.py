"""The live-mic vowel-classification demos.

Every other `DemoModule` in this app satisfies `core/demo.py`'s contract:
"everything... precomputed once, deterministically, in initialize()". A
live microphone cannot be precomputed -- it hasn't happened yet, and it
is never the same twice. Rather than bending that contract (or every
caller that trusts it) to fit these demos, `LiveVowelSnnDemoBase` keeps it
technically: `_build_frames` returns exactly one static placeholder
`Frame`, so `DemoModule.initialize()`'s checks all pass trivially, and the
SIDEBAR and FRAME-CAPTION machinery treat it like any other demo. The
actual live behavior lives entirely in `start_capture`/`stop_capture`/
`poll_and_advance`/`snapshot`, which `main_window.py` calls directly
instead of going through `StepPlayer` (see `DemoModule.supports_live_capture`
and `_select_demo`).

Two concrete demos share this one base -- same capture, same model, same
inference -- and differ only in which picture of it `main_window.py` shows:
`LiveVowelSnnDemo` is the time-over-neurons spike raster + confidence bars;
`LiveVowelSnnNodesDemo` (right after it in the sidebar) is the network
itself -- input/hidden/output neurons as nodes, lighting up with recent
activity, joined by the actual learned weights. `snapshot()` is a superset
dict both widgets read from, so adding the second view needed no second
copy of the audio/inference pipeline -- only a second renderer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from efficient_nn_lab.core.demo import DemoModule, Frame
from efficient_nn_lab.live.audio_capture import AudioSource, MicrophoneSource
from efficient_nn_lab.live.features import log_energy_bins
from efficient_nn_lab.live.vowel_model import VOWELS, VowelSnnWeights, forward, load_weights, normalize_features

_DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "vowel_snn_weights.npz"
#: How many recent LIF time-steps the rolling raster keeps on screen.
#: 10 processed windows' worth -- enough to read as a moving raster, not
#: so much that the oldest spikes on screen are from a minute ago.
_HISTORY_STEPS_WINDOWS = 10
#: How fast the live confidence/firing-rate readouts follow a new window's
#: numbers. Low enough that one noisy window (coughing, a chair creak)
#: doesn't yank the display around; high enough to visibly react within a
#: handful of windows (~1s) once someone actually speaks.
_READOUT_EMA = 0.35


def _try_load_default_weights() -> VowelSnnWeights | None:
    if not _DEFAULT_WEIGHTS_PATH.exists():
        return None
    return load_weights(str(_DEFAULT_WEIGHTS_PATH))


class LiveVowelSnnDemoBase(DemoModule):
    """Shared capture + inference for both live vowel-classifier demos.

    Subclasses provide only `title`/`slug`/`description`; see module
    docstring for why there are two of them and why duplicating this much
    instead was rejected.
    """

    supports_live_capture = True

    def __init__(self, source: AudioSource | None = None, weights: VowelSnnWeights | None = None) -> None:
        self._source: AudioSource = source if source is not None else MicrophoneSource()
        self._weights = weights if weights is not None else _try_load_default_weights()
        self.is_capturing = False
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        if self._weights is not None:
            n_in = self._weights.w1.shape[0]
            n_hidden = self._weights.w1.shape[1]
            n_out = self._weights.w2.shape[1]
            history = _HISTORY_STEPS_WINDOWS * self._weights.t_steps
            self._hidden_history = np.zeros((n_hidden, history), dtype=bool)
            self._output_history = np.zeros((n_out, history), dtype=bool)
            self._hidden_rate = np.zeros(n_hidden)
            self._confidence = np.full(n_out, 1.0 / n_out)
            self._last_intensity = np.zeros(n_in)
        else:
            self._hidden_history = np.zeros((0, 0), dtype=bool)
            self._output_history = np.zeros((0, 0), dtype=bool)
            self._hidden_rate = np.zeros(0)
            self._confidence = np.full(len(VOWELS), 1.0 / len(VOWELS))
            self._last_intensity = np.zeros(0)
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
        history and the smoothed readouts."""
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
        self._last_intensity = normalize_features(features, weights)
        hidden_cache, output_cache = forward(features, weights, seed=None)

        t_steps = weights.t_steps
        self._hidden_history = np.concatenate([self._hidden_history[:, t_steps:], hidden_cache.spikes.T.astype(bool)], axis=1)
        self._output_history = np.concatenate([self._output_history[:, t_steps:], output_cache.spikes.T.astype(bool)], axis=1)

        hidden_rate_now = hidden_cache.spikes.mean(axis=0)
        output_rate_now = output_cache.spikes.mean(axis=0)
        self._hidden_rate = (1.0 - _READOUT_EMA) * self._hidden_rate + _READOUT_EMA * hidden_rate_now
        self._confidence = (1.0 - _READOUT_EMA) * self._confidence + _READOUT_EMA * output_rate_now

    # -- what the views need ---------------------------------------------
    def snapshot(self) -> dict[str, object]:
        """A superset dict: `LiveSpikeView` (time raster + bars) and
        `LiveNodeView` (network diagram) each read only the keys they
        need, so this one method serves both without either widget
        knowing the other exists."""
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
                "w1": None,
                "w2": None,
                "input_intensity": [],
                "hidden_rate": [],
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
            "w1": self._weights.w1,
            "w2": self._weights.w2,
            "input_intensity": self._last_intensity.tolist(),
            "hidden_rate": self._hidden_rate.tolist(),
        }


class LiveVowelSnnDemo(LiveVowelSnnDemoBase):
    title = "SNN -> Classificação de vogais ao vivo (microfone)"
    slug = "live.vowel_snn"
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


class LiveVowelSnnNodesDemo(LiveVowelSnnDemoBase):
    title = "SNN -> Classificação de vogais ao vivo (neurônios)"
    slug = "live.vowel_snn_nodes"
    description = (
        "O mesmo classificador ao vivo da demo anterior, mas mostrando os "
        "NEURÔNIOS da SNN (Spiking Neural Network, rede neural de pulso) em "
        "vez do gráfico de disparos no tempo: uma bolinha por neurônio de "
        "entrada, da camada oculta e de saída, brilhando conforme a "
        "atividade recente -- e as linhas entre elas são os pesos "
        "realmente aprendidos (W1 e W2), não um desenho ilustrativo."
    )
