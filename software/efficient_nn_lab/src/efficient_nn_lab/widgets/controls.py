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
        root.addLayout(transport_row)

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
        root.addLayout(speed_row)

        toggle_row = QHBoxLayout()
        self._eq_btn = QPushButton("Mostrar equação")
        self._eq_btn.setCheckable(True)
        self._eq_btn.toggled.connect(self.show_equation_toggled)
        self._expl_btn = QPushButton("Mostrar explicação")
        self._expl_btn.setCheckable(True)
        self._expl_btn.setChecked(True)
        self._expl_btn.toggled.connect(self.show_explanation_toggled)
        toggle_row.addWidget(self._eq_btn)
        toggle_row.addWidget(self._expl_btn)
        root.addLayout(toggle_row)

        self._params_layout = QVBoxLayout()
        root.addLayout(self._params_layout)

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

    def _on_slider(self, name: str, raw_value: int, scale: float, label: QLabel, cfg: dict) -> None:
        value = raw_value / scale
        label.setText(f"{cfg['label']}: {value:g}")
        self.parameter_changed.emit(name, value)
