"""Play a 1-D sample out loud (FIXME §8 companion — "hear the data").

The animation ▶ transport steps frames; it is not sound. This is the only place
the app produces audio: it takes whatever waveform a view is currently showing
and plays it through the default output device via ``QtMultimedia.QAudioSink``.

No resampling of the science — the PCM is emitted at the signal's own sample
rate. Amplitude is peak-normalised for listening only (meeting01 windows are
z-scored, so their raw amplitude is arbitrary); this is a ``DISPLAY_ONLY``
convenience, never fed back into the pipeline.

If QtMultimedia is missing or there is no output device, :func:`play` raises
``RuntimeError`` naming the cause and the remedy — it never fails silently.
"""

from __future__ import annotations

import numpy as np

try:  # QtMultimedia is a separate wheel/module from the rest of PySide6
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject
    from PySide6.QtMultimedia import (
        QAudioFormat,
        QAudioSink,
        QMediaDevices,
    )

    _IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    QObject = object  # type: ignore
    _IMPORT_ERROR = str(exc)


def is_available() -> bool:
    """True when the QtMultimedia module imported. Device presence is only
    checked at :meth:`AudioPlayer.play` time (querying devices needs a running
    QApplication)."""
    return not _IMPORT_ERROR


class AudioPlayer(QObject):
    """Owns one ``QAudioSink`` at a time. Keep an instance alive on the view
    (the sink + buffer must outlive the call or playback is cut off)."""

    def __init__(self, parent=None) -> None:
        if _IMPORT_ERROR:
            super().__init__()
        else:
            super().__init__(parent)
        self._sink = None
        self._buffer = None

    def stop(self) -> None:
        if self._sink is not None:
            self._sink.stop()
            self._sink = None
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def play(self, samples: np.ndarray, sample_rate: float) -> float:
        """Play ``samples`` (mono, 1-D) at ``sample_rate`` Hz. Returns the clip
        duration in seconds. Raises ``RuntimeError`` with a remedy on any
        problem."""
        if _IMPORT_ERROR:
            raise RuntimeError(
                "audio playback needs PySide6.QtMultimedia, which failed to "
                f"import ({_IMPORT_ERROR}). Install it with "
                "`./.venv/bin/pip install PySide6-Addons` (or the distro "
                "`pyside6` multimedia package)."
            )
        data = np.asarray(samples, dtype=np.float64).reshape(-1)
        if data.size == 0:
            raise RuntimeError("nothing to play — the signal is empty.")
        if not np.all(np.isfinite(data)):
            raise RuntimeError(
                "signal has NaN/Inf samples — refusing to send them to the "
                "audio device."
            )
        sr = float(sample_rate)
        if not np.isfinite(sr) or sr < 3000.0:
            raise RuntimeError(
                f"sample rate {sample_rate!r} is not audible (need >= 3000 Hz). "
                "EEG channels are not audio."
            )

        devices = QMediaDevices.audioOutputs()
        if not devices or QMediaDevices.defaultAudioOutput().isNull():
            raise RuntimeError(
                "no audio output device found. Start a sound server "
                "(PipeWire / PulseAudio) or plug in an output, then retry. "
                "In a headless session there is no device to play through."
            )

        peak = float(np.max(np.abs(data)))
        norm = data / peak * 0.92 if peak > 0 else data
        pcm = np.clip(np.round(norm * 32767.0), -32768, 32767).astype("<i2")

        fmt = QAudioFormat()
        fmt.setSampleRate(int(round(sr)))
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        dev = QMediaDevices.defaultAudioOutput()
        if not dev.isFormatSupported(fmt):
            fmt = dev.preferredFormat()
            fmt.setChannelCount(1)

        self.stop()
        self._buffer = QBuffer(self)
        self._buffer.setData(QByteArray(pcm.tobytes()))
        self._buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self._sink = QAudioSink(dev, fmt, self)
        self._sink.start(self._buffer)
        return pcm.size / float(fmt.sampleRate())
