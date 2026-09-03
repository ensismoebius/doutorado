"""Main application window: demo tree, canvas area, controls, two modes.

Wires together everything under core/ (DemoModule, StepPlayer, AppState)
and widgets/ (SignalView, WeightView, NeuronView, ControlsWidget) without
any of those modules knowing about each other — this is the one place that
does.
"""

from __future__ import annotations

import sys
from math import ceil

import numpy as np

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFontMetrics, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
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

from efficient_nn_lab.app.math_render import MathTextLabel, render_math_image
from efficient_nn_lab.app.theme import STYLESHEET, TEXT_COLOR
from efficient_nn_lab.core.animation import StepPlayer
from efficient_nn_lab.core.demo import DemoModule
from efficient_nn_lab.core.state import AppState
from efficient_nn_lab.widgets.controls import ControlsWidget
from efficient_nn_lab.widgets.neuron_view import NeuronView
from efficient_nn_lab.widgets.signal_view import SignalView
from efficient_nn_lab.widgets.weight_view import WeightView

from efficient_nn_lab.backprop.demos.chain_rule_layers import ChainRuleLayersDemo
from efficient_nn_lab.backprop.demos.matrix_algebra import MatrixAlgebraDemo
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
_NEURON_KINDS = {
    "backprop_pipeline", "mlp_network", "matrix_algebra", "chain_layers", "forward_pipeline", "ste_pipeline",
    "guided_pipeline", "comparison_pipeline",
}

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
    "R = reset, N = próxima demo, Esc = voltar ao menu."
)

# The frame-title and explanation labels under the canvas are kept at a
# fixed height per demo so their changing text never reflows the layout
# (and resizes the animation canvas) mid-playback. How much space that
# reservation needs used to be *estimated* from a characters-per-line
# guess -- which was wrong in three separate ways at once (see
# _reserve_text_heights), the visible symptom being clipped descenders on
# every step title. It is now measured. These are only the ceilings, so a
# pathological text cannot eat the whole canvas.
_EXPL_MAX_LINES = 6
_FRAME_MAX_LINES = 2
#: Fallback width for the measurement when the label has not been laid out
#: yet (a --demo deep link measures during __init__, before the first
#: layout pass). The resize hook re-measures with the real width later.
_MIN_MEASURE_WIDTH = 400
#: The animation canvas never shrinks below this, whatever the text under
#: it needs. Demos are watched, not read: a correct-but-huge caption that
#: squeezes the drawing to a strip defeats the purpose of both.
_MIN_CANVAS_HEIGHT = 330
#: ...but the floor is a floor, not a promise. Everything else in the right
#: column (title, description, the two captions, the equation frame, the
#: controls) has a minimum of its own, and if their sum plus the canvas
#: floor exceeds the window, Qt satisfies nobody and lays the widgets ON
#: TOP of each other -- which is what a fixed 330 did at the 900x600
#: minimum window: the step title and the explanation painted over the
#: drawing. So the floor is whatever is left after the chrome, capped at
#: _MIN_CANVAS_HEIGHT and never below _ABS_MIN_CANVAS_HEIGHT.
_ABS_MIN_CANVAS_HEIGHT = 140
#: ...and the two labels together may never reserve more than this share of
#: the window height. Without the cap, a narrow window wraps the
#: explanation onto more lines, the reservation grows, and it grows at the
#: canvas's expense -- exactly when the canvas is already smallest.
_MAX_TEXT_SHARE = 0.28
#: The demo's description paragraph is a subtitle, not the demo. Capped so
#: a wordy one cannot claim a third of a short window.
_DESCRIPTION_MAX_LINES = 4

# The equation panel is a fixed-height framed box so toggling it never
# reflows the right column mid-playback; the rendered equation is scaled
# to fit inside it. Rendered at a high DPI so the downscale stays crisp.
_EQUATION_FRAME_HEIGHT = 108
_EQUATION_DPI = 320


def _height_for_lines(fm: QFontMetrics, height_px: float, max_lines: int, padding: int) -> int:
    """Round a measured pixel height up to whole lines, capped and padded."""
    spacing = fm.lineSpacing()
    lines = max(1, min(max_lines, ceil(height_px / spacing)))
    return spacing * lines + padding


def _build_demo_tree() -> dict[str, list[DemoModule]]:
    return {
        "Backpropagation": [
            TraditionalBackpropDemo(),
            MultilayerNetworkDemo(),
            MatrixAlgebraDemo(),
            ChainRuleLayersDemo(),
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


def _demo_order(groups: dict[str, list[DemoModule]]) -> list[DemoModule]:
    """Every demo, in the order the sidebar tree shows them.

    This flat list *is* the lecture running order: walking it forwards
    means "next item in this section, or the first item of the next
    section once this one runs out" (FIXME.md), because the tree is built
    section by section. Keeping it derived from the same dict the tree is
    built from is what stops the two from drifting apart.
    """
    return [demo for demos in groups.values() for demo in demos]


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
        # kept for _apply_canvas_floor, which measures this column's own
        # spacing/margins instead of assuming them.
        self._right_column = right
        root.addLayout(right, stretch=1)

        top_bar = QHBoxLayout()
        self.demo_title_label = QLabel("Efficient Neural Networks Lab")
        self.demo_title_label.setObjectName("DemoTitle")
        top_bar.addWidget(self.demo_title_label, stretch=1)
        # Lecture-mode navigation (FIXME.md): lecture mode hides the demo
        # tree, which is the only way to change demo -- so in the mode
        # meant for presenting there was no way to move on without
        # leaving it. This button is that way. It is visible in both
        # modes (it is a convenience with the tree, a necessity without
        # it) and is enabled only once some demo is on screen.
        self.next_demo_btn = QPushButton("Próxima demo \u25b8")
        self.next_demo_btn.setToolTip(
            "Vai para o próximo item desta seção — ou para o primeiro da "
            "seção seguinte, se este for o último (tecla N)"
        )
        self.next_demo_btn.clicked.connect(self._on_next_demo)
        top_bar.addWidget(self.next_demo_btn)
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
        self.demo_description_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        # Read once, at the start of a demo -- unlike the animation, which
        # is looked at the whole time. At the minimum window size the full
        # paragraph took 150px straight out of the canvas, so it is capped;
        # the complete text stays available as the label's tooltip.
        self.demo_description_label.setMaximumHeight(_DESCRIPTION_MAX_LINES * QFontMetrics(
            self.demo_description_label.font()
        ).lineSpacing() + 6)
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
        # The animation is the demo; the text under it is the caption. When
        # the window is short, Qt has to take the missing pixels from
        # somewhere, and with only the labels pinned to a fixed height it
        # took them all from here -- at the 900x600 minimum the LIF canvas
        # came out 179px tall, which is the "the graphs are too small"
        # complaint all over again. _apply_canvas_floor makes the squeeze
        # land on the chrome instead, and _reserve_text_heights caps the
        # captions so the two cannot fight over the same pixels.
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
        # one line until a demo is selected (the welcome screen shows no
        # step title); _reserve_text_heights measures the real need then.
        self._set_reserved_height(self.frame_label, 0.0, 1, padding=6)
        right.addWidget(self.frame_label)

        # Off-screen twin used only to measure how tall an explanation
        # really renders (inline formula images included) -- see
        # _measure_rich_height. Never parented into the layout, so it is
        # never painted.
        self._measure_label = MathTextLabel("")
        self._measure_label.setWordWrap(True)

        self.explanation_label = MathTextLabel("")
        self.explanation_label.setObjectName("Explanation")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._set_reserved_height(self.explanation_label, 0.0, 1, padding=8)
        right.addWidget(self.explanation_label)

        self.equation_frame = QFrame()
        self.equation_frame.setObjectName("EquationFrame")
        self.equation_frame.setVisible(False)
        eq_layout = QVBoxLayout(self.equation_frame)
        eq_layout.setContentsMargins(10, 8, 10, 8)
        self.equation_label = QLabel("")
        self.equation_label.setObjectName("Equation")
        self.equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        eq_layout.addWidget(self.equation_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.equation_frame.setFixedHeight(_EQUATION_FRAME_HEIGHT)
        self._equation_pixmap: QPixmap | None = None
        right.addWidget(self.equation_frame)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("Detail")
        self.detail_label.setWordWrap(True)
        self.detail_label.setVisible(False)
        right.addWidget(self.detail_label)

        self.next_demo_btn.setEnabled(False)

        self.controls = ControlsWidget()
        self.controls.reset_clicked.connect(self._on_reset)
        self.controls.step_backward_clicked.connect(self._on_step_backward)
        self.controls.step_forward_clicked.connect(self._on_step_forward)
        self.controls.play_clicked.connect(self._on_play)
        self.controls.pause_clicked.connect(self._on_pause)
        self.controls.fast_loop_clicked.connect(self._on_fast_loop)
        self.controls.speed_changed.connect(self._on_speed_changed)
        self.controls.parameter_changed.connect(self._on_parameter_changed)
        self.controls.show_equation_toggled.connect(self._on_equation_toggled)
        self.controls.show_explanation_toggled.connect(self.explanation_label.setVisible)
        right.addWidget(self.controls)
        self.controls.setEnabled(False)

        # every sibling now exists, so the floor can be measured
        self._apply_canvas_floor()

    def _build_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._toggle_play_pause)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=self._on_step_forward)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=self._on_step_backward)
        QShortcut(QKeySequence(Qt.Key.Key_R), self, activated=self._on_reset)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._show_welcome)
        # N for "next demo" -- deliberately not PageDown/Right: Right is
        # already "next step inside this demo", and presenter remotes send
        # PageDown for "next slide", which must not jump demos.
        QShortcut(QKeySequence(Qt.Key.Key_N), self, activated=self._on_next_demo)

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
        self.demo_description_label.setToolTip(demo.description)
        self.controls.setEnabled(True)
        self.next_demo_btn.setEnabled(True)
        self.controls.rebuild_parameters(demo.parameters())
        self.controls.set_fast_loop_available(demo.supports_fast_loop)
        self._reserve_text_heights(demo)
        self._refresh_frame()

    # -- lecture-mode navigation (FIXME.md) -------------------------------
    def _on_next_demo(self) -> None:
        """Advance to the next demo in the sidebar's running order.

        From the last demo of a section this lands on the first demo of the
        next section (the flat order in _demo_order already encodes that),
        and from the very last demo it wraps around to the first. Wrapping
        is deliberate: a dead button at the end of the deck is a failure
        the presenter only discovers live, mid-talk, with no tree visible
        to fall back on.
        """
        order = _demo_order(self._demo_groups)
        if not order:
            return
        current = self.player.demo if self.player is not None else None
        if current is None:
            index = -1
        else:
            index = next((i for i, demo in enumerate(order) if demo is current), -1)
        self._go_to_demo(order[(index + 1) % len(order)])

    def _go_to_demo(self, demo: DemoModule) -> None:
        """Select ``demo`` and move the tree's highlight with it.

        The tree is hidden in lecture mode, but its current item still has
        to follow along: leaving lecture mode must not reveal a sidebar
        pointing at some demo other than the one on screen.
        """
        self._highlight_in_tree(demo)
        self._select_demo(demo)

    def _highlight_in_tree(self, demo: DemoModule) -> None:
        for i in range(self.tree.topLevelItemCount()):
            parent_item = self.tree.topLevelItem(i)
            for j in range(parent_item.childCount()):
                child = parent_item.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) is demo:
                    self.tree.setCurrentItem(child)
                    return

    def _reserve_text_heights(self, demo: DemoModule) -> None:
        """Size the fixed-height labels to *this* demo's actual text.

        The frame-title and explanation labels are fixed-height so their
        changing text never reflows the layout and resizes the animation
        canvas mid-playback. But a single global reservation sized for the
        wordiest demo wastes vertical space on every shorter one:
        SNN->LIF's explanations are all one line, yet it paid the full
        reservation which — together with its 4 parameter sliders — left
        its canvas ~80px shorter than the original layout. Reserving per
        demo keeps the no-reflow guarantee while giving each demo the
        tallest canvas its own text allows.

        The reservation is **measured**, not estimated. The previous
        estimate (``len(text) / 80`` characters per line, times the
        *explanation* label's line spacing, for both labels) was wrong
        three separate ways, and the three cancelled out just enough to
        look plausible:

        1. it sized the frame-title label with the explanation label's
           font. The title is 22pt bold (34px line) against the
           explanation's 18pt (28px), so a one-line title got 6px less
           room than one line of it needs -- the clipped descenders that
           are visible on every single step title;
        2. a ``--demo`` deep link runs this from ``__init__``, before Qt
           has applied the stylesheet, so both fonts measured as the 9pt
           default: a 24px reservation for a 34px line;
        3. the explanation label is a MathTextLabel that lays out *rich
           text* with inline formula images. Those are taller than a text
           line, so a character count cannot predict its height at all
           (measured: one bitnet.ste explanation needs 134px where the
           character estimate says 56).

        So each label is now asked how tall it really needs to be, at its
        own font and its own current width, for the wordiest text this
        demo will show. Only checkpoint frames are measured: tween frames
        reuse the text of the checkpoint they animate towards, so the set
        of distinct strings is the same and measuring 240 frames instead
        of 35 would only cost time.
        """
        expl_texts = {f.explanation or "" for f in demo.checkpoint_frames()}
        frame_texts = {
            self._frame_title_text(f.label or "", demo.total_steps, demo.total_steps)
            for f in demo.checkpoint_frames()
        }
        # The two labels share one budget: whatever the title takes, the
        # explanation cannot also take. Measuring alone is not enough --
        # a correct measurement of a wordy demo in a short window would
        # still eat the canvas (see _MAX_TEXT_SHARE / _MIN_CANVAS_HEIGHT).
        budget = max(1, int(self.height() * _MAX_TEXT_SHARE))
        title_height = self._set_reserved_height(
            self.frame_label, self._measure_plain_height(self.frame_label, frame_texts),
            _FRAME_MAX_LINES, padding=6, ceiling=budget,
        )
        self._set_reserved_height(
            self.explanation_label, self._measure_rich_height(expl_texts),
            _EXPL_MAX_LINES, padding=8, ceiling=budget - title_height,
        )

    def _label_measure_width(self, label: QLabel) -> int:
        return max(_MIN_MEASURE_WIDTH, label.width() - 6)

    def _measure_plain_height(self, label: QLabel, texts: set[str]) -> float:
        """Tallest wrapped height any of ``texts`` needs in ``label``'s font."""
        label.ensurePolished()
        fm = QFontMetrics(label.font())
        width = self._label_measure_width(label)
        rect = QRect(0, 0, width, 0)
        return max(
            [float(fm.lineSpacing())]
            + [float(fm.boundingRect(rect, Qt.TextFlag.TextWordWrap, t).height()) for t in texts if t]
        )

    def _measure_rich_height(self, texts: set[str]) -> float:
        """Tallest height the explanation label needs, formulas included.

        Measured on an off-screen twin rather than on the label itself:
        setting text on the visible label to measure it would flash every
        explanation of the demo on screen before settling on the right
        one. The twin is a MathTextLabel too, so it lays the inline
        formula images out exactly the way the real one will.
        """
        self.explanation_label.ensurePolished()
        twin = self._measure_label
        twin.setFont(self.explanation_label.font())
        width = self._label_measure_width(self.explanation_label)
        tallest = float(QFontMetrics(self.explanation_label.font()).lineSpacing())
        for text in texts:
            if not text:
                continue
            twin.set_math_text(text)
            tallest = max(tallest, float(twin.heightForWidth(width)))
        return tallest

    def _set_reserved_height(
        self, label: QLabel, needed_px: float, max_lines: int, padding: int, ceiling: int | None = None
    ) -> int:
        """Apply a measured reservation, but only when it actually changed.

        This is called from resizeEvent as well, and setting a fixed
        height triggers another layout pass -- assigning the same value
        every time would make the two chase each other.

        ``ceiling`` caps the reservation in pixels. Reaching it means the
        text will clip: that is the deliberate trade, because the
        alternative (an honest reservation for a wordy demo in a short
        window) squeezes the animation to a strip, and the animation is
        what the demo is for. One whole line is always granted, whatever
        the ceiling says. Returns the height applied.
        """
        label.ensurePolished()
        metrics = QFontMetrics(label.font())
        height = _height_for_lines(metrics, needed_px, max_lines, padding)
        if ceiling is not None:
            height = max(metrics.lineSpacing() + padding, min(height, ceiling))
        if label.height() != height or label.minimumHeight() != height:
            label.setFixedHeight(height)
        return height

    def _apply_canvas_floor(self) -> None:
        """Reserve as much height for the animation as the window can spare.

        A fixed floor is wrong in both directions: too low and the drawing
        is a strip (59px at the minimum window size, before this existed),
        too high and the column's minimums no longer fit the window, at
        which point Qt stops honouring them and lays the widgets ON TOP of
        each other -- the captions painted over the canvas.

        So the floor is measured, not guessed: whatever the window has left
        after every sibling's own minimum, capped at _MIN_CANVAS_HEIGHT.
        Measured rather than a constant because the chrome is not fixed --
        a demo with five sliders has a taller controls panel than one with
        none, and the captions' reservation changes per demo.
        """
        siblings = (
            self.demo_title_label, self.demo_description_label, self.frame_label,
            self.explanation_label, self.equation_frame, self.controls,
        )
        chrome = sum(
            max(widget.minimumSizeHint().height(), widget.minimumHeight()) for widget in siblings
        )
        spacing = self._right_column.spacing() * len(siblings)
        margins = self._right_column.contentsMargins()
        spare = self.height() - chrome - spacing - margins.top() - margins.bottom()
        floor = max(_ABS_MIN_CANVAS_HEIGHT, min(_MIN_CANVAS_HEIGHT, spare))
        if self.stack.minimumHeight() != floor:
            self.stack.setMinimumHeight(floor)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Re-measure the reservations for the new width.

        Wrapping depends on the label's width, so the number of lines a
        given explanation needs changes when the window does. Without
        this, a window resized narrower keeps the reservation computed for
        the wider one and clips the text that now wraps to more lines.
        """
        super().resizeEvent(event)
        self._apply_canvas_floor()
        if self.player is not None:
            self._reserve_text_heights(self.player.demo)

    @staticmethod
    def _frame_title_text(label: str, step: int, total: int) -> str:
        """The step-title line, in one place: shown *and* measured with it."""
        return f"{label}    (passo {step}/{total})"

    def _show_welcome(self) -> None:
        if self.player is not None:
            self.player.pause()
        self.demo_title_label.setText("Efficient Neural Networks Lab")
        self.demo_description_label.setText(_WELCOME_TEXT)
        self.frame_label.setText("")
        self.explanation_label.set_math_text("")
        self._set_equation("")
        self.controls.setEnabled(False)

    def _show_references(self) -> None:
        if self.player is not None:
            self.player.pause()
        self.demo_title_label.setText("Referências")
        self.demo_description_label.setText("")
        self.stack.setCurrentWidget(self.references_view)
        self.frame_label.setText("")
        self.explanation_label.set_math_text("")
        self._set_equation("")
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

    def _on_fast_loop(self) -> None:
        if self.player:
            self.player.play_fast_loop()

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
        # next_demo_btn stays visible on purpose: with the tree gone it is
        # the only demo-to-demo navigation left.

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

        self.frame_label.setText(
            self._frame_title_text(frame.label, demo.current_step + 1, demo.total_steps)
        )
        self.explanation_label.set_math_text(frame.explanation)
        self._set_equation(frame.equation)
        if self.state.professor_mode:
            self.detail_label.setText(_format_professor_detail(frame.values))
        self.controls.set_playing(demo.is_playing)
        self.controls.set_fast_loop_active(self.player.is_looping)

        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _on_equation_toggled(self, visible: bool) -> None:
        self.equation_frame.setVisible(visible)
        if visible:
            self._rescale_equation()

    def _set_equation(self, equation: str) -> None:
        """Render ``frame.equation`` as a big typeset equation in its frame."""
        self._equation_pixmap = None
        self.equation_label.clear()
        if not equation:
            self.equation_label.setText("(sem equação para este passo)")
            return
        img = render_math_image(equation, dpi=_EQUATION_DPI, color=TEXT_COLOR)
        if img is None or img.isNull():
            self.equation_label.setText(equation)
            return
        self._equation_pixmap = QPixmap.fromImage(img)
        self._rescale_equation()

    def _rescale_equation(self) -> None:
        """Fit the stored equation pixmap inside the fixed frame (no cropping)."""
        if self._equation_pixmap is None:
            return
        box_w = max(140, self.equation_frame.width() - 24)
        box_h = _EQUATION_FRAME_HEIGHT - 24
        pix = self._equation_pixmap
        if pix.width() > box_w or pix.height() > box_h:
            pix = pix.scaled(
                box_w,
                box_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.equation_label.setPixmap(pix)
