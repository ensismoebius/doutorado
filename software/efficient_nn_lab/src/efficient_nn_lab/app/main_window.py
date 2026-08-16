"""Main application window: demo tree, canvas area, controls, two modes.

Wires together everything under core/ (DemoModule, StepPlayer, AppState)
and widgets/ (SignalView, WeightView, NeuronView, ControlsWidget) without
any of those modules knowing about each other — this is the one place that
does.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from efficient_nn_lab.app.theme import STYLESHEET
from efficient_nn_lab.core.animation import StepPlayer
from efficient_nn_lab.core.demo import DemoModule
from efficient_nn_lab.core.state import AppState
from efficient_nn_lab.widgets.controls import ControlsWidget
from efficient_nn_lab.widgets.neuron_view import NeuronView
from efficient_nn_lab.widgets.signal_view import SignalView
from efficient_nn_lab.widgets.weight_view import WeightView

from efficient_nn_lab.backprop.demos.multilayer_network import MultilayerNetworkDemo
from efficient_nn_lab.backprop.demos.traditional_gd import TraditionalBackpropDemo
from efficient_nn_lab.bitnet.demos.backward import BackwardSTEDemo
from efficient_nn_lab.bitnet.demos.forward import ForwardLossDemo
from efficient_nn_lab.bitnet.demos.guided_sequence import GuidedBitNetDemo
from efficient_nn_lab.bitnet.demos.scalar_quantization import ScalarQuantizationDemo
from efficient_nn_lab.comparison.ann_bitnet_snn import AnnBitnetSnnComparisonDemo
from efficient_nn_lab.snn.demos.lif_dynamics import LIFDynamicsDemo
from efficient_nn_lab.snn.demos.poisson_coding import PoissonCodingDemo
from efficient_nn_lab.snn.demos.poisson_image_coding import PoissonImageCodingDemo
from efficient_nn_lab.snn.demos.spike_generation import SpikeGenerationDemo
from efficient_nn_lab.snn.demos.surrogate_gradient import SurrogateGradientDemo

#: Which widget renders which frame "kind" (see core/demo.py's Frame.values
#: — every frame in a demo carries the same "kind" tag throughout).
_SIGNAL_KINDS = {
    "signal_spikes", "poisson_spikes", "poisson_image_coding", "lif_trace", "backprop_convergence",
}
_WEIGHT_KINDS = {"scalar_quantization", "staircase", "quant_derivative", "surrogate_curve"}
_NEURON_KINDS = {"backprop_pipeline", "mlp_network", "forward_pipeline", "ste_pipeline", "guided_pipeline", "comparison_pipeline"}

_REFERENCES_TEXT = """\
Referências

Wang, H.; Ma, S.; Dong, L.; Huang, S.; Wang, H.; Ma, L.; Yang, F.; Wang, R.;
Wu, Y.; Wei, F. BitNet: Scaling 1-bit Transformers for Large Language
Models. arXiv:2310.11453, 2023.

Ma, S.; Wang, H.; Ma, L.; Wang, L.; Wang, W.; Huang, S.; Dong, L.; Wang, R.;
Xue, J.; Wei, F. The Era of 1-bit LLMs: All Large Language Models are in
1.58 Bits. arXiv:2402.17764, 2024.

Neftci, E. O.; Mostafa, H.; Zenke, F. Surrogate Gradient Learning in
Spiking Neural Networks. IEEE Signal Processing Magazine, v. 36, n. 6,
p. 51-63, 2019. DOI: 10.1109/MSP.2019.2931595.

Metadados verificados por resolução de DOI / busca antes da inclusão nesta
tela (ver ESPECIFICACAO_DLVL.md #33).
"""

_WELCOME_TEXT = (
    "Efficient Neural Networks Lab\n\n"
    "Selecione uma demonstração à esquerda para começar.\n"
    "Atalhos: Espaço = play/pause, -> = próximo passo, <- = passo anterior, "
    "R = reset, Esc = voltar ao menu."
)


def _build_demo_tree() -> dict[str, list[DemoModule]]:
    return {
        "Backpropagation": [
            TraditionalBackpropDemo(),
            MultilayerNetworkDemo(),
        ],
        "BitNet": [
            ScalarQuantizationDemo(),
            ForwardLossDemo(),
            BackwardSTEDemo(),
            GuidedBitNetDemo(),
        ],
        "SNN": [
            SpikeGenerationDemo(),
            PoissonCodingDemo(),
            PoissonImageCodingDemo(),
            LIFDynamicsDemo(),
            SurrogateGradientDemo(),
        ],
        "Comparação": [
            AnnBitnetSnnComparisonDemo(),
        ],
    }


def _choose_view(values: dict[str, object]) -> str:
    kind = values.get("kind")
    if kind in _SIGNAL_KINDS:
        return "signal"
    if kind in _WEIGHT_KINDS:
        return "weight"
    if kind in _NEURON_KINDS:
        return "neuron"
    # tween frames mid-transition between two different "kind"s (a
    # deliberate scene cut, see build_sequence's steps=0 gaps) briefly
    # have neither populated; keep showing whichever kind is closer to
    # being fully revealed so the cut still lands on the right widget.
    return "neuron"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Efficient Neural Networks Lab")
        self.resize(1180, 720)
        self.setStyleSheet(STYLESHEET)

        self.state = AppState()
        self._demo_groups = _build_demo_tree()
        self.player: StepPlayer | None = None

        self._build_ui()
        self._build_shortcuts()

    # -- UI construction --------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMaximumWidth(260)
        for category, demos in self._demo_groups.items():
            parent_item = QTreeWidgetItem([category])
            self.tree.addTopLevelItem(parent_item)
            for demo in demos:
                child = QTreeWidgetItem([demo.title])
                child.setData(0, Qt.ItemDataRole.UserRole, demo)
                parent_item.addChild(child)
            parent_item.setExpanded(True)
        ref_item = QTreeWidgetItem(["Referências"])
        ref_item.setData(0, Qt.ItemDataRole.UserRole, "references")
        self.tree.addTopLevelItem(ref_item)
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        root.addWidget(self.tree)

        right = QVBoxLayout()
        root.addLayout(right, stretch=1)

        top_bar = QHBoxLayout()
        self.demo_title_label = QLabel("Efficient Neural Networks Lab")
        self.demo_title_label.setObjectName("DemoTitle")
        top_bar.addWidget(self.demo_title_label, stretch=1)
        self.lecture_mode_btn = QPushButton("Modo palestra")
        self.lecture_mode_btn.setCheckable(True)
        self.lecture_mode_btn.toggled.connect(self._on_lecture_mode_toggled)
        self.professor_mode_btn = QPushButton("Modo professor")
        self.professor_mode_btn.setCheckable(True)
        self.professor_mode_btn.toggled.connect(self._on_professor_mode_toggled)
        top_bar.addWidget(self.lecture_mode_btn)
        top_bar.addWidget(self.professor_mode_btn)
        right.addLayout(top_bar)

        self.demo_description_label = QLabel(_WELCOME_TEXT)
        self.demo_description_label.setWordWrap(True)
        right.addWidget(self.demo_description_label)

        self.stack = QStackedWidget()
        self.signal_view = SignalView()
        self.weight_view = WeightView()
        self.neuron_view = NeuronView()
        self.references_view = QTextEdit(_REFERENCES_TEXT)
        self.references_view.setReadOnly(True)
        self.stack.addWidget(self.signal_view)
        self.stack.addWidget(self.weight_view)
        self.stack.addWidget(self.neuron_view)
        self.stack.addWidget(self.references_view)
        right.addWidget(self.stack, stretch=1)

        self.frame_label = QLabel("")
        self.frame_label.setObjectName("FrameTitle")
        right.addWidget(self.frame_label)

        self.explanation_label = QLabel("")
        self.explanation_label.setObjectName("Explanation")
        self.explanation_label.setWordWrap(True)
        right.addWidget(self.explanation_label)

        self.equation_label = QLabel("")
        self.equation_label.setStyleSheet("font-family: monospace; color: #444;")
        self.equation_label.setVisible(False)
        right.addWidget(self.equation_label)

        self.detail_label = QLabel("")
        self.detail_label.setStyleSheet("font-family: monospace; font-size: 9pt; color: #555;")
        self.detail_label.setWordWrap(True)
        self.detail_label.setVisible(False)
        right.addWidget(self.detail_label)

        self.controls = ControlsWidget()
        self.controls.reset_clicked.connect(self._on_reset)
        self.controls.step_backward_clicked.connect(self._on_step_backward)
        self.controls.step_forward_clicked.connect(self._on_step_forward)
        self.controls.play_clicked.connect(self._on_play)
        self.controls.pause_clicked.connect(self._on_pause)
        self.controls.speed_changed.connect(self._on_speed_changed)
        self.controls.parameter_changed.connect(self._on_parameter_changed)
        self.controls.show_equation_toggled.connect(self.equation_label.setVisible)
        self.controls.show_explanation_toggled.connect(self.explanation_label.setVisible)
        right.addWidget(self.controls)
        self.controls.setEnabled(False)

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._on_step_forward)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self._on_step_backward)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, activated=self._on_reset)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._show_welcome)

    # -- demo selection -----------------------------------------------
    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        if payload == "references":
            self._show_references()
        elif isinstance(payload, DemoModule):
            self._select_demo(payload)

    def _select_demo(self, demo: DemoModule) -> None:
        demo.reset()
        if self.player is not None:
            self.player.pause()
            self.player.frame_changed.disconnect(self._refresh_frame)
            self.player.playback_finished.disconnect(self._on_playback_finished)
        self.player = StepPlayer(demo)
        self.player.frame_changed.connect(self._refresh_frame)
        self.player.playback_finished.connect(self._on_playback_finished)

        self.demo_title_label.setText(demo.title)
        self.demo_description_label.setText(demo.description)
        self.controls.setEnabled(True)
        self.controls.rebuild_parameters(demo.parameters())
        self._refresh_frame()

    def _show_welcome(self) -> None:
        if self.player is not None:
            self.player.pause()
        self.demo_title_label.setText("Efficient Neural Networks Lab")
        self.demo_description_label.setText(_WELCOME_TEXT)
        self.frame_label.setText("")
        self.explanation_label.setText("")
        self.controls.setEnabled(False)

    def _show_references(self) -> None:
        if self.player is not None:
            self.player.pause()
        self.demo_title_label.setText("Referências")
        self.demo_description_label.setText("")
        self.stack.setCurrentWidget(self.references_view)
        self.frame_label.setText("")
        self.explanation_label.setText("")
        self.controls.setEnabled(False)

    # -- control callbacks --------------------------------------------
    def _on_reset(self) -> None:
        if self.player:
            self.player.reset()

    def _on_step_backward(self) -> None:
        if self.player:
            self.player.step_backward()

    def _on_step_forward(self) -> None:
        if self.player:
            self.player.step_forward()

    def _on_play(self) -> None:
        if self.player:
            self.player.play()

    def _on_pause(self) -> None:
        if self.player:
            self.player.pause()

    def _toggle_play_pause(self) -> None:
        if not self.player:
            return
        if self.player.demo.is_playing:
            self.player.pause()
        else:
            self.player.play()

    def _on_speed_changed(self, speed: float) -> None:
        if self.player:
            self.player.set_speed(speed)

    def _on_parameter_changed(self, name: str, value: float) -> None:
        if not self.player:
            return
        self.player.pause()
        self.player.demo.set_parameter(name, value)
        self._refresh_frame()

    def _on_playback_finished(self) -> None:
        self.controls.set_playing(False)

    # -- mode toggles -----------------------------------------------------
    def _on_lecture_mode_toggled(self, enabled: bool) -> None:
        self.state.lecture_mode = enabled
        self.tree.setVisible(not enabled)
        self.professor_mode_btn.setVisible(not enabled)

    def _on_professor_mode_toggled(self, enabled: bool) -> None:
        self.state.professor_mode = enabled
        self.detail_label.setVisible(enabled)
        if enabled:
            self.controls.show_equation_toggled.emit(True)
        self._refresh_frame()

    # -- rendering ------------------------------------------------------
    def _refresh_frame(self) -> None:
        if self.player is None:
            return
        demo = self.player.demo
        frame = demo.current_frame()

        view_name = _choose_view(frame.values)
        view = {"signal": self.signal_view, "weight": self.weight_view, "neuron": self.neuron_view}[view_name]
        self.stack.setCurrentWidget(view)
        view.render(frame.values)

        self.frame_label.setText(f"{frame.label}    (passo {demo.current_step + 1}/{demo.total_steps})")
        self.explanation_label.setText(frame.explanation)
        self.equation_label.setText(frame.equation or "(sem equação para este passo)")
        if self.state.professor_mode:
            self.detail_label.setText("estado completo: " + repr(frame.values))
        self.controls.set_playing(demo.is_playing)
