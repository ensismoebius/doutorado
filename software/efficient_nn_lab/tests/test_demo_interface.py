"""Contract test: every DemoModule in the app must obey the same rules,
regardless of what it demonstrates (ESPECIFICACAO_DLVL.md #4).
"""

import pytest

from efficient_nn_lab.bitnet.demos.backward import BackwardSTEDemo
from efficient_nn_lab.bitnet.demos.forward import ForwardLossDemo
from efficient_nn_lab.bitnet.demos.guided_sequence import GuidedBitNetDemo
from efficient_nn_lab.bitnet.demos.scalar_quantization import ScalarQuantizationDemo
from efficient_nn_lab.comparison.ann_bitnet_snn import AnnBitnetSnnComparisonDemo
from efficient_nn_lab.snn.demos.lif_dynamics import LIFDynamicsDemo
from efficient_nn_lab.snn.demos.poisson_coding import PoissonCodingDemo
from efficient_nn_lab.snn.demos.spike_generation import SpikeGenerationDemo
from efficient_nn_lab.snn.demos.surrogate_gradient import SurrogateGradientDemo

ALL_DEMO_CLASSES = [
    ScalarQuantizationDemo,
    ForwardLossDemo,
    BackwardSTEDemo,
    GuidedBitNetDemo,
    SpikeGenerationDemo,
    PoissonCodingDemo,
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
