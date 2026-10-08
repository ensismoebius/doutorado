"""Application-wide UI state that is not owned by any single demo.

Kept as a small Qt object (not a plain dataclass) because the main window
and the controls bar both need to react live to these flags (professor mode
toggling detail panels, lecture mode simplifying the chrome) — a Qt signal
is the natural way to broadcast that without wiring every widget together
by hand.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AppState(QObject):
    """Cross-cutting UI toggles.

    professor_mode: show numeric values, equations, gradients, internal
    parameters (ESPECIFICACAO_DLVL.md #29). Default off — a lecture
    audience sees the intuitive view first.

    lecture_mode: collapses the tree/menu chrome into the two-button
    Anterior/Proximo layout described in ESPECIFICACAO_DLVL.md #28, driven
    by keyboard shortcuts instead of mouse navigation.

    show_bitnet: include the BitNet demos (sidebar "BitNet" group, plus
    the BitNet column of the ANN x BitNet x SNN comparison) in the
    running deck. Default on, matching the app's original scope. Off is
    for presenting just the thesis' own material (redes de pulso +
    lógica paraconsistente) without the unrelated BitNet frontier that
    `documentation/08-lectures/fronteiras-bitnets-redes-pulso/` also
    covers -- see that talk's `apresentacao-tese.tex` for the slide-deck
    half of the same split.
    """

    professor_mode_changed = Signal(bool)
    lecture_mode_changed = Signal(bool)
    show_bitnet_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._professor_mode = False
        self._lecture_mode = False
        self._show_bitnet = True

    @property
    def professor_mode(self) -> bool:
        return self._professor_mode

    @professor_mode.setter
    def professor_mode(self, value: bool) -> None:
        if value != self._professor_mode:
            self._professor_mode = value
            self.professor_mode_changed.emit(value)

    @property
    def lecture_mode(self) -> bool:
        return self._lecture_mode

    @lecture_mode.setter
    def lecture_mode(self, value: bool) -> None:
        if value != self._lecture_mode:
            self._lecture_mode = value
            self.lecture_mode_changed.emit(value)

    @property
    def show_bitnet(self) -> bool:
        return self._show_bitnet

    @show_bitnet.setter
    def show_bitnet(self, value: bool) -> None:
        if value != self._show_bitnet:
            self._show_bitnet = value
            self.show_bitnet_changed.emit(value)
