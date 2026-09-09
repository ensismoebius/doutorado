import numpy as np
import pytest

pytest.importorskip("pyqtgraph")

from experiment_microscope.core.integrity import Origin
from experiment_microscope.data.adapters import FeatureMatrix, TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.triangle_view import TriangleView


class _FakeThesis:
    def __init__(self):
        self._fm = FeatureMatrix(
            values=np.random.default_rng(0).standard_normal((5, 8)),
            feature_names=tuple(f"f{j}" for j in range(8)),
            sample_labels=tuple(f"s{i}" for i in range(5)),
            class_labels=(1, 1, 2, 2, 3),
            origin=Origin.COMPUTED,
            set_label="fake / handcrafted-lfcc",
        )

    def load_features(self, node):
        return self._fm

    def paraconsistent_points(self):
        return []

    def _summary(self, phase, tag):
        return {"modality": "eeg", "handcrafted": {"wavelet": "haar", "dtwpt_level": 4}}


def _repo():
    r = DataRepository()
    r._adapters["thesis"] = _FakeThesis()
    return r


def test_triangle_ignores_non_run(qapp):
    v = TriangleView(_repo())
    v.show_node(TreeNode("sample", "s", {"level": "sample"}), "thesis")
    assert v._run_node is None


def test_triangle_binds_run_and_sets_sample_range(qapp):
    v = TriangleView(_repo())
    node = TreeNode("run", "r", {"level": "run", "phase": "phase00", "run_tag": "t"})
    v.show_node(node, "thesis")
    assert v._run_node is node
    assert v._sample.maximum() == 4  # 5 samples
    v._sample.setValue(3)  # scrubbing must not raise
    assert "sample 3/4" in v._status.text()
