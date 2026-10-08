"""Audio sources feeding the live vowel demo: a real microphone, or a
synthetic stand-in for tests and the training script.

`MicrophoneSource` wraps `sounddevice.InputStream`. PortAudio calls its
callback on its OWN thread, never Qt's -- so that callback does the one
thing it is safe to do from another thread: push raw samples into a plain
`queue.Queue`. Nothing here ever touches a Qt widget; `MainWindow`'s own
timer (on the Qt thread) is what drains the queue and redraws.
"""

from __future__ import annotations

import queue
from typing import Protocol

import numpy as np


class AudioSource(Protocol):
    """What `LiveVowelSnnDemo` needs from any audio source, real or fake."""

    sample_rate: int

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def read_available(self) -> np.ndarray: ...


class MicrophoneSource:
    """Live microphone input via `sounddevice`, one mono float32 stream."""

    def __init__(self, sample_rate: int = 16000, block_size: int = 512) -> None:
        self.sample_rate = sample_rate
        self._block_size = block_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001 -- sounddevice's own signature
        # Runs on PortAudio's thread. `indata` is reused by PortAudio after
        # this call returns, so it must be copied before queuing.
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        import sounddevice as sd  # imported lazily: only needed with a real mic

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self._block_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        # Drop anything left unread -- a demo re-started later should not
        # replay stale audio from before it was stopped.
        with self._queue.mutex:
            self._queue.queue.clear()

    def read_available(self) -> np.ndarray:
        chunks = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


class SyntheticSource:
    """Same interface as `MicrophoneSource`, feeding pre-supplied samples.

    Used by tests and by `train.py`/`record.py`'s own smoke-checks so
    nothing in this app's test suite ever has to open a real microphone
    (there usually isn't one to open in CI, and even where there is, a
    test that depends on real ambient sound is not a repeatable test).
    `read_available` hands out `chunk_size` samples at a time, mimicking a
    real stream's incremental delivery rather than returning everything at
    once.
    """

    def __init__(self, samples: np.ndarray, sample_rate: int = 16000, chunk_size: int = 512) -> None:
        self.sample_rate = sample_rate
        self._samples = np.asarray(samples, dtype=np.float32)
        self._chunk_size = chunk_size
        self._pos = 0
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read_available(self) -> np.ndarray:
        if not self._running:
            return np.zeros(0, dtype=np.float32)
        end = min(self._pos + self._chunk_size, len(self._samples))
        chunk = self._samples[self._pos : end]
        self._pos = end
        return chunk
