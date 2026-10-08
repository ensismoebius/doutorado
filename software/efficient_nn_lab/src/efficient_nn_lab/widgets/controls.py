"""Playback and parameter controls shared by every demo.

(ESPECIFICACAO_DLVL.md #25: every module gets Reset/Step/Play/Pause, a
speed control, its own relevant parameters, and "Show equation"/"Show
explanation" toggles — parameters only appear when the active demo
actually exposes them.)

This widget never touches a DemoModule or StepPlayer directly; it only
emits signals. main_window.py is the one place that knows how those
signals map onto the current demo, which keeps this widget reusable
across every demo without any per-demo special-casing here.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class ControlsWidget(QWidget):
    reset_clicked = Signal()
    step_backward_clicked = Signal()
    step_forward_clicked = Signal()
    play_clicked = Signal()
    pause_clicked = Signal()
    fast_loop_clicked = Signal()
    capture_toggled = Signal(bool)
    speed_changed = Signal(float)
    parameter_changed = Signal(str, float)
    show_equation_toggled = Signal(bool)
    show_explanation_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._param_sliders: dict[str, QSlider] = {}
        self._param_labels: dict[str, QLabel] = {}
        self._param_scale: dict[str, float] = {}
        self._is_playing = False

        root = QVBoxLayout(self)

        # Reset/Step/Play/speed share one container so the live-capture
        # demo (which has none of those -- see set_transport_visible) can
        # hide them all with a single setVisible call, the same pattern
        # self._params_container already uses for the sliders area.
        self._transport_container = QWidget()
        transport_container_layout = QVBoxLayout(self._transport_container)
        transport_container_layout.setContentsMargins(0, 0, 0, 0)

        transport_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Voltar ao início da demonstração (R)")
        self._back_btn = QPushButton("<- Anterior")
        self._back_btn.setToolTip("Passo anterior (←)")
        # a single toggling button, not separate Play/Pause buttons: it
        # reads "Play" while paused and "Pause" while playing, and clicking
        # it always does the opposite of whatever is currently happening.
        self._play_btn = QPushButton("Play")
        self._play_btn.setToolTip("Reproduzir / pausar a animação (Espaço)")
        self._fwd_btn = QPushButton("Proximo ->")
        self._fwd_btn.setToolTip("Próximo passo (→)")
        # Opt-in per demo (DemoModule.supports_fast_loop): hidden unless the
        # active demo is one whose cadence is itself the point. Checkable so
        # the control shows *that a continuous mode is on*, which a momentary
        # button could not; its checked state is mirrored from the player by
        # set_fast_loop_active(), never tracked here.
        self._loop_btn = QPushButton("Loop rápido")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setToolTip(
            "Roda todos os passos em sequência, sem pausa e em ciclo contínuo — "
            "a cadência em que o olho integra os disparos e a imagem aparece"
        )
        self._loop_btn.setVisible(False)
        for btn in (self._reset_btn, self._back_btn, self._play_btn, self._fwd_btn, self._loop_btn):
            transport_row.addWidget(btn)
        transport_container_layout.addLayout(transport_row)

        self._loop_btn.clicked.connect(self._on_loop_clicked)

        self._reset_btn.clicked.connect(self.reset_clicked)
        self._back_btn.clicked.connect(self.step_backward_clicked)
        self._fwd_btn.clicked.connect(self.step_forward_clicked)
        self._play_btn.clicked.connect(self._on_play_pause_clicked)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Velocidade:"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(25, 300)
        self._speed_slider.setValue(100)
        self._speed_slider.valueChanged.connect(lambda v: self.speed_changed.emit(v / 100.0))
        speed_row.addWidget(self._speed_slider)
        transport_container_layout.addLayout(speed_row)
        root.addWidget(self._transport_container)

        # Opt-in per demo (DemoModule.supports_live_capture): hidden unless
        # the active demo is the one driven by a real microphone instead
        # of the usual Reset/Step/Play transport (see set_transport_visible,
        # which hides that whole transport for exactly this demo).
        capture_row = QHBoxLayout()
        self._capture_btn = QPushButton("Iniciar captura ao vivo")
        self._capture_btn.setCheckable(True)
        self._capture_btn.setToolTip("Liga/desliga a captura do microfone para este demo")
        self._capture_btn.setVisible(False)
        self._capture_btn.toggled.connect(self._on_capture_toggled)
        capture_row.addWidget(self._capture_btn)
        root.addLayout(capture_row)

        toggle_row = QHBoxLayout()
        self._eq_btn = QPushButton("Mostrar equação")
        self._eq_btn.setCheckable(True)
        self._eq_btn.toggled.connect(self.show_equation_toggled)
        self._expl_btn = QPushButton("Mostrar explicação")
        self._expl_btn.setCheckable(True)
        self._expl_btn.setChecked(True)
        self._expl_btn.toggled.connect(self.show_explanation_toggled)
        # Starts unchecked/hidden on purpose: the sliders are for someone
        # who already wants to poke at a parameter, not something every
        # viewer needs on screen for every demo. Shown only for a demo
        # that actually HAS parameters (rebuild_parameters below), same
        # rule set_fast_loop_available already applies to the loop button.
        self._params_btn = QPushButton("Mostrar parâmetros")
        self._params_btn.setCheckable(True)
        self._params_btn.setVisible(False)
        self._params_btn.toggled.connect(self._on_params_toggled)
        toggle_row.addWidget(self._eq_btn)
        toggle_row.addWidget(self._expl_btn)
        toggle_row.addWidget(self._params_btn)
        root.addLayout(toggle_row)

        self._params_layout = QVBoxLayout()
        # One container around the whole params_layout, not a per-slider
        # setVisible(False): rebuild_parameters tears down and rebuilds
        # the individual slider rows on every demo switch, so hiding them
        # one by one would have to be redone every time too. Hiding the
        # single container instead is one setVisible call, and it is
        # exactly what toggled() below already does.
        self._params_container = QWidget()
        self._params_container.setLayout(self._params_layout)
        self._params_container.setVisible(False)
        root.addWidget(self._params_container)

    def _on_params_toggled(self, visible: bool) -> None:
        self._params_container.setVisible(visible)

    def set_playing(self, playing: bool) -> None:
        self._is_playing = playing
        self._play_btn.setText("Pause" if playing else "Play")

    def set_fast_loop_available(self, available: bool) -> None:
        """Show the loop control only for demos that offer it."""
        self._loop_btn.setVisible(available)
        if not available:
            self.set_fast_loop_active(False)

    def set_fast_loop_active(self, active: bool) -> None:
        """Mirror the player's loop state onto the button.

        Signals are blocked while doing it: this is called from the
        per-frame refresh, and letting setChecked() re-emit would feed the
        state straight back into the handler that caused it.
        """
        if self._loop_btn.isChecked() == active:
            return
        self._loop_btn.blockSignals(True)
        self._loop_btn.setChecked(active)
        self._loop_btn.blockSignals(False)

    def set_transport_visible(self, visible: bool) -> None:
        """Hide Reset/Step/Play/speed entirely for a live-capture demo,
        which has no notion of "steps" to play/pause/step through."""
        self._transport_container.setVisible(visible)

    def set_live_capture_available(self, available: bool) -> None:
        """Show the capture button only for the one demo that has it."""
        self._capture_btn.setVisible(available)
        if not available:
            self.set_capture_active(False)

    def set_capture_active(self, active: bool) -> None:
        """Mirror the demo's actual capture state onto the button without
        re-emitting capture_toggled -- same reasoning as set_fast_loop_active."""
        if self._capture_btn.isChecked() == active:
            return
        self._capture_btn.blockSignals(True)
        self._capture_btn.setChecked(active)
        self._capture_btn.blockSignals(False)
        self._capture_btn.setText("Parar captura" if active else "Iniciar captura ao vivo")

    def _on_capture_toggled(self, checked: bool) -> None:
        self._capture_btn.setText("Parar captura" if checked else "Iniciar captura ao vivo")
        self.capture_toggled.emit(checked)

    def _on_loop_clicked(self) -> None:
        # A second click on an active loop means "stop", which is exactly
        # Pause -- no separate stop path to keep in sync.
        if self._loop_btn.isChecked():
            self.fast_loop_clicked.emit()
        else:
            self.pause_clicked.emit()

    def _on_play_pause_clicked(self) -> None:
        if self._is_playing:
            self.pause_clicked.emit()
        else:
            self.play_clicked.emit()

    def rebuild_parameters(self, spec: dict[str, dict[str, object]]) -> None:
        """Rebuild the parameter sliders for the active demo.

        ``spec`` matches DemoModule.parameters(): name -> {label, min, max,
        step, value}. Sliders are integer-only, so real-valued parameters
        are scaled by 1/step internally and unscaled before emitting
        parameter_changed.
        """
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_sliders.clear()
        self._param_labels.clear()
        self._param_scale.clear()

        for name, cfg in spec.items():
            row = QHBoxLayout()
            label = QLabel(f"{cfg['label']}: {cfg['value']}")
            slider = QSlider(Qt.Orientation.Horizontal)
            step = float(cfg.get("step", 0.1)) or 0.1
            scale = 1.0 / step
            slider.setRange(int(cfg["min"] * scale), int(cfg["max"] * scale))
            slider.setValue(int(cfg["value"] * scale))
            slider.valueChanged.connect(lambda v, n=name, s=scale, lb=label, cfg=cfg: self._on_slider(n, v, s, lb, cfg))
            row.addWidget(label)
            row.addWidget(slider)
            container = QWidget()
            container.setLayout(row)
            self._params_layout.addWidget(container)
            self._param_sliders[name] = slider
            self._param_labels[name] = label
            self._param_scale[name] = scale

        has_params = bool(spec)
        self._params_btn.setVisible(has_params)
        if not has_params:
            # no button left to show it with -- e.g. switching from a demo
            # that had parameters (toggled open) to one that has none must
            # not leave an empty, visible params area behind.
            self._params_btn.setChecked(False)
            self._params_container.setVisible(False)
        else:
            self._params_container.setVisible(self._params_btn.isChecked())

    def _on_slider(self, name: str, raw_value: int, scale: float, label: QLabel, cfg: dict) -> None:
        value = raw_value / scale
        label.setText(f"{cfg['label']}: {value:g}")
        self.parameter_changed.emit(name, value)
