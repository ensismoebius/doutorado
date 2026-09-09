"""One general animation controller for the whole application (FIXME §24).

Rather than each temporal view owning its own timer, every animatable view
(spike propagation, membrane potential, wavelet decomposition, reconstruction,
latent trajectory) registers a frame count and a per-frame callback with the
single ``TimelinePlayer``. The transport bar drives this one object.

The player only ever emits a frame *index*; it never computes anything. Views
pull the frame's data from the ``TransformationCache`` (FIXME §24: "should not
recompute expensive transformations on every frame").
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

_BASE_TICK_MS = 40  # ~25 fps
_SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)  # FIXME §17


class TimelinePlayer(QObject):
    """Shared transport: Play / Pause / Stop / Step / Loop / Speed (FIXME §24)."""

    frame_changed = Signal(int)        # current frame index
    range_changed = Signal(int)        # total frame count changed
    playback_finished = Signal()
    playing_changed = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._total = 1
        self._frame = 0
        self._speed = 1.0
        self._loop = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._in_tick = False

    # -- configuration -------------------------------------------------
    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def current_frame(self) -> int:
        return self._frame

    @property
    def is_playing(self) -> bool:
        return self._timer.isActive()

    @property
    def is_looping(self) -> bool:
        return self._loop

    @property
    def speed(self) -> float:
        return self._speed

    def set_total_frames(self, total: int) -> None:
        total = max(1, int(total))
        if total == self._total:
            return
        self._total = total
        if self._frame >= total:
            self._frame = total - 1
            self.frame_changed.emit(self._frame)
        self.range_changed.emit(total)

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.1, float(speed))
        if self._timer.isActive():
            self._timer.setInterval(self._interval_ms())

    def set_looping(self, value: bool) -> None:
        self._loop = bool(value)

    def _interval_ms(self) -> int:
        return max(5, int(_BASE_TICK_MS / self._speed))

    # -- transport ---------------------------------------------------
    def play(self) -> None:
        if self._total <= 1:
            return
        if self._frame >= self._total - 1:
            self._frame = 0
            self.frame_changed.emit(0)
        self._timer.start(self._interval_ms())
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playing_changed.emit(False)

    def stop(self) -> None:
        self.pause()
        self.seek(0)

    def step_forward(self) -> None:
        self.pause()
        self.seek(self._frame + 1)

    def step_backward(self) -> None:
        self.pause()
        self.seek(self._frame - 1)

    def seek(self, frame: int) -> None:
        frame = max(0, min(self._total - 1, int(frame)))
        if frame != self._frame:
            self._frame = frame
            self.frame_changed.emit(frame)

    def _advance(self) -> None:
        if self._in_tick:
            return
        self._in_tick = True
        try:
            nxt = self._frame + 1
            if nxt >= self._total:
                if self._loop:
                    self.seek(0)
                else:
                    self.pause()
                    self.playback_finished.emit()
                return
            self.seek(nxt)
        finally:
            self._in_tick = False


SPEED_PRESETS = _SPEEDS
