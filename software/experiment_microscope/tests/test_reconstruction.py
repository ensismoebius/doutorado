import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.data.adapters import LatentTrace, TreeNode
from experiment_microscope.data.meeting01_adapter import Meeting01Adapter, _recon_metrics
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.reconstruction_view import ReconstructionView


def test_recon_metrics_perfect_and_offset():
    a = np.linspace(-1, 1, 64)
    perfect = _recon_metrics(a, a.copy())
    assert perfect["mse"].magnitude == 0.0
    assert abs(perfect["r2"].magnitude - 1.0) < 1e-9
    off = _recon_metrics(a, a + 0.5)
    assert off["mae"].magnitude == pytest.approx(0.5)


def test_load_latent_without_models_raises_with_remedy(tmp_path):
    a = Meeting01Adapter(results_dir=tmp_path)
    node = TreeNode("sample", "w", {"level": "window", "dataset": "fsdd", "cv_fold": 0,
                                    "split": "test", "row": 0})
    with pytest.raises(RuntimeError) as ei:
        a.load_latent(node)
    assert "save_models" in str(ei.value) and "*_encoder.npz" in str(ei.value)


class _FakeAdapter:
    def load_latent(self, node):
        rng = np.random.default_rng(0)
        orig = rng.standard_normal(80)
        return LatentTrace(
            latent=rng.standard_normal(8), reconstruction=orig + 0.1 * rng.standard_normal(80),
            original=orig, spikes=None, v_mem=None, origin=Origin.COMPUTED,
            metrics=_recon_metrics(orig, orig),
        )


def test_view_renders_trace(qapp):
    repo = DataRepository()
    repo._adapters["meeting01"] = _FakeAdapter()
    v = ReconstructionView(repo)
    v.show_node(TreeNode("sample", "w", {"level": "window"}), "meeting01")
    assert v._trace is not None
    assert v.can_export()
    assert "latent dim 8" in v._status.text()


def test_real_snn_ae_forward_end_to_end(qapp, first_fsdd_window):
    """When Step E .npz models exist, load_latent runs snn_ae_forward for real."""
    from experiment_microscope.processing._binding import is_available
    if not is_available():
        pytest.skip("nn_microscope not built")
    node, adapter = first_fsdd_window()
    if node is None or not adapter._snn_model_specs():
        pytest.skip("no FSDD corpus or no trained *_encoder.npz on disk")
    trace = adapter.load_latent(node)
    assert trace.latent.ndim == 1 and trace.latent.size > 0
    assert trace.reconstruction.shape == trace.original.shape
    assert not trace.metrics["mse"].is_missing
    v = ReconstructionView(DataRepository())
    v._trace = trace
    v._render()
    assert v.can_export()


def test_view_shows_remedy_when_adapter_raises(qapp):
    class _Raiser:
        def load_latent(self, node):
            raise RuntimeError("run a LOSO fold with save_models: true")

    repo = DataRepository()
    repo._adapters["meeting01"] = _Raiser()
    v = ReconstructionView(repo)
    v.show_node(TreeNode("sample", "w", {"level": "window"}), "meeting01")
    assert v._trace is None
    assert "save_models" in v._status.text()
