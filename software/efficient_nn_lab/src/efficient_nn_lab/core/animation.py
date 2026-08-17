"""Qt-side driver that turns a DemoModule's frames into actual playback.

This is the piece that makes "Step" and "Play" feel continuous instead of
jump-cutting between pictures: both drive the same fast frame-advance
timer through every tween frame between two checkpoints, so a single
"Próximo" click plays a short, smooth transition rather than teleporting.
Play chains those transitions with a short readable dwell at each
checkpoint before continuing — long enough to read the explanation text
that just changed, short enough not to feel like a slideshow.

DemoModule itself (core/demo.py) has no Qt dependency and stays unit
testable; this class owns the only QTimer in the picture.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal

from efficient_nn_lab.core.demo import DemoModule

#: Interval between fine-grained frame advances during any animated
#: transition. ~40ms is a comfortable ~25fps for matplotlib redraws.
_TICK_MS = 40
#: How long Play dwells on a checkpoint before animating to the next one,
#: at normal (1.0) speed — enough time to read a one-sentence explanation.
_DWELL_MS = 1100
#: Interval between frames in continuous-loop mode (see play_fast_loop).
#: Deliberately the same as _TICK_MS rather than something smaller: the
#: heaviest demo that uses this mode (snn.poisson_image) needs ~39ms of
#: measured p90 redraw time per frame, so a shorter interval would only
#: queue redraws the widget cannot service and make the UI sluggish. The
#: speed-up over Play comes from dropping the per-checkpoint dwell
#: entirely (30 checkpoints x 1100ms = 33s a lap, versus ~1.2s here), not
#: from a faster tick.
_LOOP_TICK_MS = _TICK_MS


class StepPlayer(QObject):
    """Drives ``demo`` through its frames, always by animating, never by
    jumping — see module docstring."""

    frame_changed = Signal()
    playback_finished = Signal()

    def __init__(self, demo: DemoModule, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.demo = demo
        self.speed = 1.0  # 1.0 = normal, <1 slower, >1 faster
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._dwell_timer = QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._on_dwell_finished)
        # "step_forward" | "step_backward" | "play" | "loop" | None
        self._mode: str | None = None
        # Re-entrancy guard for _tick. frame_changed triggers a full widget
        # redraw, and that re-enters the Qt event loop -- so this timer can
        # fire *inside* a tick that is still running and advance the demo a
        # second time. Measured: on snn.poisson_image the redraw costs ~39ms
        # against a 40ms interval, so the two are close enough for the
        # nesting to actually happen (a probe advancing 8 frames by hand
        # executed 16 ticks). Without this, playback speed silently doubles
        # under a slow renderer, and deep nesting is possible if it gets
        # slower still.
        self._in_tick = False
        self._target_frame_index: int = 0

    def set_demo(self, demo: DemoModule) -> None:
        self.pause()
        self.demo = demo
        self.frame_changed.emit()

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, speed)
        if self._timer.isActive():
            self._timer.setInterval(self._interval_ms())

    @property
    def is_looping(self) -> bool:
        """True while continuous-loop playback is running.

        The controls widget mirrors this onto its "Loop rápido" button, so
        the button's checked state is always derived from the player rather
        than tracked separately (one source of truth, no way to drift).
        """
        return self._mode == "loop"

    def _interval_ms(self) -> int:
        base = _LOOP_TICK_MS if self._mode == "loop" else _TICK_MS
        return max(5, int(base / self.speed))

    def _dwell_ms(self) -> int:
        return max(150, int(_DWELL_MS / self.speed))

    # -- public transport controls --------------------------------------
    def play(self) -> None:
        self._dwell_timer.stop()
        self.demo.play()
        self._mode = "play"
        self._start_leg_to(self.demo.target_frame_index_forward())

    def play_fast_loop(self) -> None:
        """Play continuously and without dwelling, wrapping at the end.

        For demos where the *cadence* is the lesson: every frame is shown
        back to back, and reaching the last one rewinds to the first
        instead of stopping, so the sequence reads as a moving picture
        rather than a slideshow. Only Pause/Reset (or leaving the demo)
        stops it.
        """
        self._dwell_timer.stop()
        self.demo.play()
        self._mode = "loop"
        self._timer.start(self._interval_ms())
        # so the UI reflects "looping" immediately instead of only on the
        # first tick (pause/reset emit for the same reason)
        self.frame_changed.emit()

    def pause(self) -> None:
        self._timer.stop()
        self._dwell_timer.stop()
        self._mode = None
        self.demo.pause()
        self.frame_changed.emit()

    def reset(self) -> None:
        self._timer.stop()
        self._dwell_timer.stop()
        self._mode = None
        self.demo.reset()
        self.frame_changed.emit()

    def step_forward(self) -> None:
        if self.demo.current_step >= self.demo.total_steps - 1:
            return
        self._dwell_timer.stop()
        self._mode = "step_forward"
        self._start_leg_to(self.demo.target_frame_index_forward())

    def step_backward(self) -> None:
        if self.demo.current_step <= 0 and self.demo.is_at_first_frame():
            return
        self._dwell_timer.stop()
        self._mode = "step_backward"
        self._start_leg_to(self.demo.target_frame_index_backward())

    # -- internal state machine ------------------------------------------
    def _start_leg_to(self, target_frame_index: int) -> None:
        self._target_frame_index = target_frame_index
        if self.demo.current_frame_index == target_frame_index:
            self._on_leg_finished()
            return
        self._timer.start(self._interval_ms())

    def _tick(self) -> None:
        if self._in_tick:
            return
        self._in_tick = True
        try:
            self._tick_once()
        finally:
            self._in_tick = False

    def _tick_once(self) -> None:
        if self._mode == "loop":
            # advance_one_frame() clears is_playing at the end of the
            # sequence, which is right for Play and wrong here: rewind and
            # keep going, so the lap boundary is invisible.
            if self.demo.is_at_last_frame():
                self.demo.rewind_to_start()
                self.demo.play()
            else:
                self.demo.advance_one_frame()
            self.frame_changed.emit()
            return

        if self._mode == "step_backward":
            self.demo.retreat_one_frame()
        else:
            self.demo.advance_one_frame()
        self.frame_changed.emit()

        reached = (
            self.demo.current_frame_index == self._target_frame_index
            or self.demo.is_at_last_frame()
            or self.demo.is_at_first_frame()
        )
        if reached:
            self._timer.stop()
            self._on_leg_finished()

    def _on_leg_finished(self) -> None:
        if self._mode == "play":
            if self.demo.is_at_last_frame():
                self._mode = None
                self.demo.pause()
                self.playback_finished.emit()
            else:
                self._dwell_timer.start(self._dwell_ms())
        else:
            self._mode = None

    def _on_dwell_finished(self) -> None:
        if self._mode == "play" and self.demo.is_playing:
            self._start_leg_to(self.demo.target_frame_index_forward())
