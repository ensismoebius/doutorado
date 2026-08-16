"""Main application window: demo tree, canvas area, controls, two modes.

Wires together everything under core/ (DemoModule, StepPlayer, AppState)
and widgets/ (SignalView, WeightView, NeuronView, ControlsWidget) without
any of those modules knowing about each other — this is the one place that
does.
"""

from __future__ import annotations

import sys

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics, QKeySequence, QShortcut
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


def _summarize_value(value: object) -> str:
    """One-line, numbers-not-noise rendering of a frame value.

    Scalars keep their float formatting; long numeric sequences (numpy
    arrays, or plain lists/tuples of numbers) are shown as
    "first3, ..., last3" with their shape/count instead of being dumped
    verbatim -- which is what makes repr() unreadable for demos whose
    frames carry whole trace arrays.
    """
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        shape = value.shape
        n = flat.size
        if n == 0:
            return f"array({shape}, vazio)"
        if n <= 8:
            body = ", ".join(f"{x:.4g}" for x in flat)
        else:
            body = ", ".join(f"{x:.4g}" for x in flat[:3]) + ", ..., " + ", ".join(f"{x:.4g}" for x in flat[-3:])
        return f"[{body}] — shape {shape}, {n} valores"
    if isinstance(value, (list, tuple)) and value and all(isinstance(x, (int, float)) for x in value):
        n = len(value)
        if n <= 8:
            body = ", ".join(f"{x:.4g}" for x in value)
        else:
            body = ", ".join(f"{x:.4g}" for x in value[:3]) + ", ..., " + ", ".join(f"{x:.4g}" for x in value[-3:])
        return f"[{body}] — {n} valores"
    if isinstance(value, float):
        return f"{value:.4g}"
    return repr(value)


def _format_professor_detail(values: dict[str, object]) -> str:
    return "\n".join(f"{key}: {_summarize_value(val)}" for key, val in values.items())
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

# The frame-title and explanation labels under the canvas are kept at a
# fixed height per demo so their changing text never reflows the layout
# (and resizes the animation canvas) mid-playback. These control how much
# space is reserved: a deliberately conservative characters-per-line
# estimate and a ceiling in lines. See _reserve_text_heights.
_CHARS_PER_LINE = 80
_EXPL_MAX_LINES = 4
_FRAME_MAX_LINES = 2


def _lines_for(text: str, max_lines: int) -> int:
    """Number of wrapped lines ``text`` needs, clamped to ``max_lines``."""
    if not text:
        return 1
    needed = (len(text) + _CHARS_PER_LINE - 1) // _CHARS_PER_LINE
    return max(1, min(max_lines, needed))


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
    def __init__(self, initial_demo_slug: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Efficient Neural Networks Lab")
        self.resize(1180, 720)
        # Matplotlib's Qt canvas rescales its own dpi to the widget's pixel
        # size on resize, so content never literally clips off-canvas (see
        # tests/test_widgets_no_clipping.py) -- but below this floor the
        # tree/explanation/controls chrome itself gets too cramped to use,
        # and every diagram becomes too small to read regardless of not
        # being "clipped". This is a usability floor, not a clipping fix.
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLESHEET)

        self.state = AppState()
        self._demo_groups = _build_demo_tree()
        self.player: StepPlayer | None = None

        self._build_ui()
        self._build_shortcuts()

        if initial_demo_slug is not None:
            self._select_demo_by_slug(initial_demo_slug)

    # -- deep-linking (documentation/08-lectures/.../presentation.md #4) --
    def _select_demo_by_slug(self, slug: str) -> None:
        """Jump straight to a demo by its stable `DemoModule.slug`.

        Used by main.py's `--demo` flag, itself invoked from the lecture
        slides' "open in software" links. A slug that doesn't match any
        demo (typo, renamed/removed demo) is reported to stderr and
        otherwise ignored -- the app must still open normally rather than
        crash or silently do nothing, since this can be triggered live
        during a talk.
        """
        for i in range(self.tree.topLevelItemCount()):
            parent_item = self.tree.topLevelItem(i)
            for j in range(parent_item.childCount()):
                child = parent_item.child(j)
                demo = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(demo, DemoModule) and demo.slug == slug:
                    self.tree.setCurrentItem(child)
                    self._select_demo(demo)
                    return
        print(f"efficient-nn-lab: unknown --demo slug {slug!r}, ignoring", file=sys.stderr)

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

        # frame_label (the step name) and explanation_label (its didactic
        # text) change on every single animation frame -- their length
        # varies a lot between e.g. "Iteração 2 -- A perda (loss)" and a
        # full three-sentence explanation. Left to size-to-content, a
        # longer/shorter text reflows this QVBoxLayout and resizes
        # self.stack (stretch=1), which visibly resizes/shifts the
        # animation canvas above them on every frame change. Fixing their
        # height up front (sized from font metrics for a generous number
        # of lines, not the current text) reserves constant space so
        # content changes never touch the layout above. Top-aligned so
        # short text doesn't recentre within the reserved box either.
        self.frame_label = QLabel("")
        self.frame_label.setObjectName("FrameTitle")
        self.frame_label.setWordWrap(True)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.frame_label.setFixedHeight(QFontMetrics(self.frame_label.font()).lineSpacing() * _FRAME_MAX_LINES + 6)
        right.addWidget(self.frame_label)

        self.explanation_label = QLabel("")
        self.explanation_label.setObjectName("Explanation")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.explanation_label.setFixedHeight(QFontMetrics(self.explanation_label.font()).lineSpacing() * _EXPL_MAX_LINES + 8)
        right.addWidget(self.explanation_label)

        self.equation_label = QLabel("")
        self.equation_label.setObjectName("Equation")
        self.equation_label.setVisible(False)
        right.addWidget(self.equation_label)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("Detail")
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
        self._reserve_text_heights(demo)
        self._refresh_frame()

    def _reserve_text_heights(self, demo: DemoModule) -> None:
        """Size the fixed-height labels to *this* demo's actual text.

        The frame-title and explanation labels are fixed-height so their
        changing text never reflows the layout and resizes the animation
        canvas mid-playback. But a single global reservation sized for the
        wordiest demo (4 explanation lines) wastes vertical space on every
        shorter one: SNN->LIF's explanations are all one line, yet it paid
        the full 4-line reservation which — together with its 4 parameter
        sliders — left its canvas ~80px shorter than the original layout.
        Reserving per demo keeps the no-reflow guarantee while giving each
        demo the tallest canvas its own text allows.
        """
        fm = QFontMetrics(self.explanation_label.font())
        spacing = fm.lineSpacing()
        expl_lines = max(_lines_for(f.explanation or "", _EXPL_MAX_LINES) for f in demo._frames)
        frame_lines = max(_lines_for(f.label or "", _FRAME_MAX_LINES) for f in demo._frames)
        self.explanation_label.setFixedHeight(spacing * expl_lines + 8)
        self.frame_label.setFixedHeight(spacing * frame_lines + 6)

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
            self.detail_label.setText(_format_professor_detail(frame.values))
        self.controls.set_playing(demo.is_playing)
