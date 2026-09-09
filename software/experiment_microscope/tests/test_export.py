import numpy as np
import pytest

pytest.importorskip("matplotlib")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.data.adapters import FeatureMatrix, Signal1D, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.viz.mpl_export import new_figure, save_figure


def test_save_figure_rejects_unknown_format(tmp_path):
    fig = new_figure()
    fig.add_subplot(111).plot([0, 1, 2])
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "x.tiff")


@pytest.mark.parametrize("ext", [".png", ".pdf", ".svg"])
def test_save_figure_writes_each_format(tmp_path, ext):
    fig = new_figure()
    fig.add_subplot(111).plot([3, 1, 4, 1, 5])
    out = save_figure(fig, tmp_path / f"fig{ext}")
    assert out.is_file() and out.stat().st_size > 0


class _Adapter:
    def __init__(self, payload):
        self._p = payload

    def load_signal(self, node):
        return self._p if isinstance(self._p, Signal1D) else (_ for _ in ()).throw(NotImplementedError())

    def load_features(self, node):
        return self._p if isinstance(self._p, FeatureMatrix) else (_ for _ in ()).throw(NotImplementedError())


def _repo(payload):
    r = DataRepository()
    r._adapters["thesis"] = _Adapter(payload)
    return r


def test_signal_view_exports_png(qapp, tmp_path):
    from experiment_microscope.views.signal_view import SignalView

    sig = Signal1D(samples=np.sin(np.linspace(0, 9, 128)), sample_rate=100.0,
                   origin=Origin.MEASURED, label="demo")
    v = SignalView(_repo(sig))
    assert not v.can_export()
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    assert v.can_export()
    out = v.export_figure(tmp_path / "sig.png")
    assert out.is_file()


def test_feature_matrix_exports_pdf(qapp, tmp_path):
    from experiment_microscope.views.feature_matrix import FeatureMatrixView

    fm = FeatureMatrix(values=np.random.default_rng(0).standard_normal((4, 5)),
                       feature_names=tuple("abcde"), sample_labels=tuple("wxyz"),
                       class_labels=(1, 1, 2, 2), origin=Origin.COMPUTED, set_label="m")
    v = FeatureMatrixView(_repo(fm))
    v.show_node(TreeNode("run", "r", {"level": "run"}), "thesis")
    out = v.export_figure(tmp_path / "fm.pdf")
    assert out.is_file()
