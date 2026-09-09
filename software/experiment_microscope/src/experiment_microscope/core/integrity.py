"""Scientific-integrity primitives (FIXME §33, §55).

Two jobs:

1. Tag every value the GUI shows with where it came from, so a downsampled
   curve is never mistaken for the analytical data and a missing metric is
   never rendered as ``0``.
2. Refuse to emit inferential vocabulary ("best", "converged", ...) in
   generated captions unless the underlying artifact established that
   condition. This reuses the banned-word discipline that
   ``software/nn/scripts/pipeline/meeting01/monitor.py`` enforces on its own
   plain-text dashboard.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any


class Origin(enum.Enum):
    """Provenance class of a displayed value."""

    #: Read directly, unmodified, from a persisted artifact.
    MEASURED = "measured"
    #: Deterministically recomputed from source data via the C++ core
    #: (nn_microscope) — same implementation the experiment used.
    COMPUTED = "computed"
    #: A dimensionality reduction (PCA / t-SNE / ...). Not a measurement.
    PROJECTED = "projected"
    #: A decimated / downsampled representation for rendering only. The
    #: numerical inspector must never read a value from this.
    DISPLAY_ONLY = "display_only"
    #: Explicitly an estimate (e.g. paraconsistentGA UNCALIBRATED latency).
    ESTIMATED = "estimated"
    #: No value is available. Renders as an em dash, never as 0 / NaN / "".
    MISSING = "missing"


_MISSING_GLYPH = "—"  # em dash


@dataclass(frozen=True)
class Value:
    """A scalar plus its provenance and (optional) unit.

    ``Value.missing()`` is the only correct way to represent "no data" —
    constructing ``Value(0.0, Origin.MEASURED)`` for an absent metric is
    exactly the substitution FIXME §33 forbids.
    """

    magnitude: Any
    origin: Origin
    unit: str = ""
    note: str = ""

    @classmethod
    def missing(cls, note: str = "") -> "Value":
        return cls(None, Origin.MISSING, note=note)

    @property
    def is_missing(self) -> bool:
        if self.origin is Origin.MISSING or self.magnitude is None:
            return True
        return isinstance(self.magnitude, float) and math.isnan(self.magnitude)

    def display(self) -> str:
        if self.is_missing:
            return _MISSING_GLYPH
        if isinstance(self.magnitude, float):
            text = f"{self.magnitude:.6g}"
        else:
            text = str(self.magnitude)
        return f"{text} {self.unit}".strip() if self.unit else text

    def labelled(self) -> str:
        """Value with a trailing provenance tag, for inspector panels."""
        if self.is_missing:
            return _MISSING_GLYPH
        return f"{self.display()}  [{self.origin.value}]"


#: Words that assert a conclusion the raw numbers do not carry on their own.
#: Mirrors monitor.py's own banned list.
BANNED_INFERENTIAL_WORDS = (
    "best",
    "optimal",
    "significant",
    "significantly",
    "converged",
    "convergence",
    "overfitted",
    "overfitting",
    "generalizing",
    "generalises",
    "generalizes",
    "improved",
    "better",
    "worse",
    "outperforms",
)


class InferentialLanguageError(AssertionError):
    """Raised when a generated caption uses a banned inferential word."""


def assert_no_inferential_language(text: str, *, context: str = "") -> str:
    """Return ``text`` unchanged, or raise if it makes an unearned claim.

    Call this on any caption the GUI *generates*. It is not applied to text
    quoted verbatim from an artifact (that is a MEASURED string).
    """

    lowered = text.lower()
    for word in BANNED_INFERENTIAL_WORDS:
        # word-boundary check without importing re for one call
        idx = lowered.find(word)
        while idx != -1:
            before = lowered[idx - 1] if idx > 0 else " "
            after = lowered[idx + len(word)] if idx + len(word) < len(lowered) else " "
            if not before.isalnum() and not after.isalnum():
                where = f" ({context})" if context else ""
                raise InferentialLanguageError(
                    f"generated caption uses inferential word {word!r}{where}: {text!r}. "
                    f"State the observed quantity instead, or quote the artifact verbatim."
                )
            idx = lowered.find(word, idx + 1)
    return text
