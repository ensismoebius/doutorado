"""Shared contract every demonstration implements.

Two kinds of frames make up a demo's sequence:

* **checkpoints** — the didactically-named steps ("Passo 3 — quantização"),
  what ``current_step``/``total_steps`` count and what "Anterior"/"Próximo"
  jump between;
* **tweens** — many fine-grained frames interpolated between two
  consecutive checkpoints (see :func:`transition`), which is what actually
  gets *played* when moving from one checkpoint to the next, so nothing
  ever jumps straight from one picture to an unrelated one.

Everything is still precomputed once, deterministically, from the current
parameters in ``initialize()`` (no randomness — ESPECIFICACAO_DLVL.md
#35). Stepping only moves a pointer into that precomputed list; the actual
smooth playback (timing, easing, dwell time at checkpoints) is owned by
core/animation.py's StepPlayer, which is the only Qt-coupled piece. This
file stays framework-agnostic and independently testable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass, field

from efficient_nn_lab.core.math_utils import tween_values

#: Default number of interior tween frames generated between two
#: checkpoints by :func:`transition`. Each tween frame costs a full
#: matplotlib redraw (~100ms+ measured on modest hardware, dominated by
#: glyph shaping for the many live-updating numeric labels -- see
#: widgets/_mpl_perf.py), so this count trades total transition wall-clock
#: time directly against motion granularity. 8 is still enough points for
#: the smoothstep easing (core/math_utils.ease_in_out) to read as a clear,
#: settled motion, at ~a third less real time than the previous 14.
DEFAULT_TWEEN_STEPS = 8


@dataclass
class Frame:
    """One point in a demo's sequence — either a checkpoint or a tween.

    ``label`` is a short caption; ``values`` holds the numeric/categorical
    state a widget needs to draw this frame; ``explanation`` is the
    didactic text for the explanation panel; ``equation`` is the optional
    formula shown when "Mostrar equação" is toggled. Tween frames normally
    reuse the label/explanation/equation of the checkpoint they are
    animating *towards*, so the on-screen text updates the moment the
    motion starts rather than only once it finishes.
    """

    label: str
    values: dict[str, object] = field(default_factory=dict)
    explanation: str = ""
    equation: str = ""
    is_checkpoint: bool = True


def transition(
    checkpoint_a: Frame,
    checkpoint_b: Frame,
    steps: int = DEFAULT_TWEEN_STEPS,
    hold: tuple[str, ...] = (),
) -> list[Frame]:
    """Interior tween frames animating from ``checkpoint_a`` to ``checkpoint_b``.

    Returns ``steps`` frames (not including either endpoint — the caller
    already has both checkpoints in its frame list). Each tween frame
    carries checkpoint_b's label/explanation/equation, so the didactic
    text is already the "arriving" one while the motion plays.

    ``hold`` names value keys that must *not* be interpolated during the
    tween: those fields keep checkpoint_a's value on every tween frame and
    only become checkpoint_b's once the destination checkpoint itself
    appears. This is the "only update when the step shows up" pattern —
    used by backprop/demos/traditional_gd.py to keep the sigmoid inset's
    activation point and derivative (tangent) pinned to the last completed
    step instead of gliding toward the next one before that step is shown.
    """
    frames = []
    for i in range(1, steps + 1):
        t = i / (steps + 1)
        values = tween_values(checkpoint_a.values, checkpoint_b.values, t)
        for key in hold:
            values[key] = checkpoint_a.values[key]
        frames.append(
            Frame(
                label=checkpoint_b.label,
                values=values,
                explanation=checkpoint_b.explanation,
                equation=checkpoint_b.equation,
                is_checkpoint=False,
            )
        )
    return frames


def build_sequence(
    checkpoints: list[Frame],
    steps: int | list[int] = DEFAULT_TWEEN_STEPS,
    hold: tuple[str, ...] = (),
) -> list[Frame]:
    """Interleave tween frames between consecutive checkpoints.

    The common case for a demo's ``_build_frames``: build the list of
    named checkpoints, then call this once instead of hand-writing the
    tween frames between every pair. ``steps`` is either a single count
    used for every gap, or one count per gap (``len(checkpoints) - 1``
    values) — pass ``0`` for a gap that deliberately jumps to a different
    kind of scene (e.g. from a function-curve view to a block-diagram
    view) rather than trying to blend two structurally unrelated pictures.

    ``hold`` is forwarded to :func:`transition` for every gap: named value
    keys stay pinned to the departure checkpoint through the tween and
    only update when the destination checkpoint appears.
    """
    if not checkpoints:
        raise ValueError("build_sequence requires at least one checkpoint")
    if isinstance(steps, int):
        steps_per_gap = [steps] * (len(checkpoints) - 1)
    else:
        steps_per_gap = list(steps)
        if len(steps_per_gap) != len(checkpoints) - 1:
            raise ValueError("steps list must have len(checkpoints) - 1 entries")
    sequence = [checkpoints[0]]
    for a, b, n in zip(checkpoints, checkpoints[1:], steps_per_gap):
        sequence.extend(transition(a, b, n, hold=hold))
        sequence.append(b)
    return sequence


class DemoModule(ABC):
    """Base class for every demonstration in the lab.

    Subclasses implement :meth:`_build_frames`, returning the full,
    deterministic frame sequence (checkpoints + tweens) for the demo's
    *current* parameters — typically via :func:`build_sequence`.
    Everything else (navigation, checkpoint bookkeeping) lives here.
    """

    #: Short name shown in the demo tree (should mirror the slide title —
    #: see ESPECIFICACAO_DLVL.md #42).
    title: str = ""
    #: One-paragraph description shown above the animation area.
    description: str = ""
    #: Stable identifier for deep-linking from outside the app (the
    #: lecture slides' "open this demo" links, see main.py's --demo flag
    #: and documentation/08-lectures/fronteiras-bitnets-redes-pulso/
    #: presentation.md). Unlike `title`, this never changes even if the
    #: displayed title's wording does, so a link baked into a slide stays
    #: valid across later renames.
    slug: str = ""

    def __init__(self) -> None:
        self._frames: list[Frame] = []
        self._checkpoint_frame_indices: list[int] = []
        self.current_frame_index: int = 0
        self.is_playing: bool = False
        self.initialize()

    # -- parameters -------------------------------------------------
    def set_parameter(self, name: str, value: object) -> None:
        """Update a demo parameter and rebuild the frame sequence."""
        if not hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} has no parameter {name!r}")
        setattr(self, name, value)
        self.initialize()

    def parameters(self) -> dict[str, dict[str, object]]:
        """Describe the sliders/controls this demo exposes.

        Returns a mapping ``name -> {"label", "min", "max", "step",
        "value"}``. The generic controls widget builds sliders from this
        without needing to know about any specific demo.
        """
        return {}

    # -- lifecycle ----------------------------------------------------
    @abstractmethod
    def _build_frames(self) -> list[Frame]:
        """Recompute the full frame sequence (checkpoints + tweens)."""

    def initialize(self) -> None:
        self._frames = self._build_frames()
        if not self._frames:
            raise ValueError(f"{type(self).__name__} produced zero frames")
        if not self._frames[0].is_checkpoint or not self._frames[-1].is_checkpoint:
            raise ValueError(f"{type(self).__name__}: first and last frame must be checkpoints")
        self._checkpoint_frame_indices = [i for i, f in enumerate(self._frames) if f.is_checkpoint]
        self.current_frame_index = 0

    def reset(self) -> None:
        self.is_playing = False
        self.initialize()

    # -- checkpoint navigation (used by manual Step/Anterior/Proximo) --
    @property
    def total_steps(self) -> int:
        """Number of *checkpoints* — what "Passo X/Y" counts."""
        return len(self._checkpoint_frame_indices)

    @property
    def current_step(self) -> int:
        """Index of the checkpoint at-or-just-before the current frame."""
        pos = bisect_right(self._checkpoint_frame_indices, self.current_frame_index) - 1
        return max(0, pos)

    def target_frame_index_forward(self) -> int:
        """Frame index of the next checkpoint (or the last frame)."""
        next_checkpoint = min(self.current_step + 1, self.total_steps - 1)
        return self._checkpoint_frame_indices[next_checkpoint]

    def target_frame_index_backward(self) -> int:
        """Frame index of the previous checkpoint (or the first frame)."""
        prev_checkpoint = max(self.current_step - 1, 0)
        return self._checkpoint_frame_indices[prev_checkpoint]

    def step_forward(self) -> None:
        """Jump straight to the next checkpoint (no animation — see
        StepPlayer for the animated version used by the GUI)."""
        if self.current_step < self.total_steps - 1:
            self.current_frame_index = self.target_frame_index_forward()
        else:
            self.is_playing = False

    def step_backward(self) -> None:
        """Jump straight to the previous checkpoint."""
        self.current_frame_index = self.target_frame_index_backward()

    # -- fine-grained navigation (used by StepPlayer to animate) --------
    def advance_one_frame(self) -> None:
        if self.current_frame_index < len(self._frames) - 1:
            self.current_frame_index += 1
        else:
            self.is_playing = False

    def retreat_one_frame(self) -> None:
        if self.current_frame_index > 0:
            self.current_frame_index -= 1

    def is_at_last_frame(self) -> bool:
        return self.current_frame_index >= len(self._frames) - 1

    def is_at_first_frame(self) -> bool:
        return self.current_frame_index <= 0

    # -- play/pause bookkeeping (StepPlayer owns the actual timer) -----
    def play(self) -> None:
        if self.is_at_last_frame():
            self.current_frame_index = 0
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    # -- frame access -----------------------------------------------
    def current_frame(self) -> Frame:
        return self._frames[self.current_frame_index]

    def checkpoint_frames(self) -> list[Frame]:
        """The named checkpoints only, in order — handy for tests and for
        anything that wants "the 10 steps" without the tween frames."""
        return [self._frames[i] for i in self._checkpoint_frame_indices]

    def is_at_last_step(self) -> bool:
        return self.current_step >= self.total_steps - 1
