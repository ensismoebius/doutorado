"""Contract test: every DemoModule in the app must obey the same rules,
regardless of what it demonstrates (ESPECIFICACAO_DLVL.md #4).
"""

import pytest

from efficient_nn_lab.backprop.demos.matrix_algebra import MatrixAlgebraDemo
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

#: Every demo the app ships. Kept in sync with the app's own tree by
#: test_all_demo_classes_covers_the_app_tree below -- the three backprop
#: demos were missing from this list, so the whole "Backpropagation"
#: group silently escaped every contract test in this file.
ALL_DEMO_CLASSES = [
    TraditionalBackpropDemo,
    MultilayerNetworkDemo,
    MatrixAlgebraDemo,
    ScalarQuantizationDemo,
    ForwardLossDemo,
    BackwardSTEDemo,
    GuidedBitNetDemo,
    SpikeGenerationDemo,
    PoissonCodingDemo,
    PoissonImageCodingDemo,
    LIFDynamicsDemo,
    SurrogateGradientDemo,
    AnnBitnetSnnComparisonDemo,
]


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_demo_has_title_and_description(demo_cls):
    demo = demo_cls()
    assert demo.title
    assert demo.description


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_demo_starts_at_first_frame(demo_cls):
    demo = demo_cls()
    assert demo.current_step == 0
    assert demo.total_steps >= 1
    assert demo.is_playing is False


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_step_forward_does_not_exceed_last_frame(demo_cls):
    demo = demo_cls()
    for _ in range(demo.total_steps + 5):
        demo.step_forward()
    assert demo.current_step == demo.total_steps - 1


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_step_backward_does_not_go_below_zero(demo_cls):
    demo = demo_cls()
    for _ in range(5):
        demo.step_backward()
    assert demo.current_step == 0


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_reset_is_deterministic(demo_cls):
    demo = demo_cls()
    total_before = demo.total_steps
    demo.step_forward()
    demo.step_forward()
    demo.reset()
    assert demo.current_step == 0
    assert demo.total_steps == total_before


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_every_frame_is_renderable_state(demo_cls):
    demo = demo_cls()
    for _ in range(demo.total_steps):
        frame = demo.current_frame()
        assert frame.label
        assert isinstance(frame.values, dict)
        demo.step_forward()


@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_play_from_last_step_restarts(demo_cls):
    demo = demo_cls()
    for _ in range(demo.total_steps):
        demo.step_forward()
    assert demo.is_at_last_step()
    demo.play()
    assert demo.current_step == 0
    assert demo.is_playing is True


# -- the list above must not drift away from what the app actually ships ----

def test_all_demo_classes_covers_the_app_tree():
    """Every demo registered in the app must be covered by this file.

    The contract tests here are parametrized over ALL_DEMO_CLASSES, so a
    demo missing from that list is not "untested with a warning" -- it is
    silently unverified. This closes that hole: adding a demo to the app
    without adding it here fails.
    """
    from efficient_nn_lab.app.main_window import _build_demo_tree

    shipped = {type(d) for group in _build_demo_tree().values() for d in group}
    listed = set(ALL_DEMO_CLASSES)
    assert shipped - listed == set(), f"demos shipped but not contract-tested: {shipped - listed}"
    assert listed - shipped == set(), f"contract-tested but not shipped: {listed - shipped}"


def test_every_demo_is_documented_in_the_readme():
    """ESPECIFICACAO_DLVL.md #43: "documentação explicar cada demonstração".

    The README's "Demonstrações" table is that documentation, and #5 asks
    each demo to answer one stated question -- which is exactly that
    table's middle column. A demo absent from it has no stated question.
    """
    import pathlib

    from efficient_nn_lab.app.main_window import _build_demo_tree

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    missing = []
    for group in _build_demo_tree().values():
        for demo in group:
            # the table writes titles with a real arrow; demo titles use "->"
            if demo.title.replace(" -> ", " → ") not in text:
                missing.append(demo.title)
    assert not missing, f"demos absent from README's table: {missing}"


def test_every_demo_is_reachable_from_the_lecture_deck():
    """ESPECIFICACAO_DLVL.md #42: the deck and the software are one system.

    Each demo must be reachable from the slides -- either a full
    ``\\DemoSlide`` frame or a ``\\abrirNoSoftware`` footer link. A demo the
    slides never mention cannot be opened during the talk, which is the
    only context this software exists for. ``backprop.matrix`` was in
    exactly that state when it was first written.

    Skipped when the deck is not alongside (the lab is usable on its own),
    so this never turns into a failure about a missing directory.
    """
    import pathlib
    import re

    from efficient_nn_lab.app.main_window import _build_demo_tree

    deck = (
        pathlib.Path(__file__).resolve().parents[3]
        / "documentation" / "08-lectures" / "fronteiras-bitnets-redes-pulso"
    )
    if not deck.is_dir():
        pytest.skip("lecture deck not present alongside the software")

    tex = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [*deck.glob("*.tex"), *(deck / "slides").glob("*.tex")]
    )
    linked = set(re.findall(r"\\(?:DemoSlide|abrirNoSoftware)\{.*?\}\{([^}]*)\}", tex))

    # No exemptions: every demo the app ships is reachable from the deck.
    # bitnet.guided used to be exempt (documented as a spare "bonus" demo);
    # it now carries an \abrirNoSoftware link on bitnetTreinamento.tex, so
    # the rule is unconditional and stays that way.
    shipped = {d.slug for group in _build_demo_tree().values() for d in group}
    unreachable = shipped - linked
    assert not unreachable, f"demos no slide can open: {sorted(unreachable)}"

    dangling = linked - shipped
    assert not dangling, f"slides link slugs that no demo provides: {sorted(dangling)}"


# -- opt-in continuous playback ----------------------------------------

@pytest.mark.parametrize("demo_cls", ALL_DEMO_CLASSES)
def test_supports_fast_loop_is_declared_and_boolean(demo_cls):
    assert isinstance(demo_cls().supports_fast_loop, bool)


def test_fast_loop_is_opt_in_and_off_by_default():
    """Default off: flying past steps whose text must be read is noise.

    Only snn.poisson_image opts in today -- it is the one demo where the
    cadence itself carries the lesson (a single Poisson time-step is
    indistinguishable from noise; the picture only appears once frames go
    by fast enough for the eye to integrate them).
    """
    from efficient_nn_lab.core.demo import DemoModule

    assert DemoModule.supports_fast_loop is False
    opted_in = {c().slug for c in ALL_DEMO_CLASSES if c().supports_fast_loop}
    assert opted_in == {"snn.poisson_image"}, opted_in


def test_rewind_to_start_keeps_playing_and_does_not_rebuild():
    """rewind_to_start is what makes the loop seamless; reset() cannot do it.

    reset() rebuilds every frame and clears is_playing, so a loop built on
    it would stall after one lap and redo the work each time round.
    """
    demo = PoissonImageCodingDemo()
    frames_before = demo._frames
    demo.play()
    demo.current_frame_index = len(demo._frames) - 1

    demo.rewind_to_start()
    assert demo.current_frame_index == 0
    assert demo.is_playing, "rewind must not stop playback"
    assert demo._frames is frames_before, "rewind must not rebuild the frames"
