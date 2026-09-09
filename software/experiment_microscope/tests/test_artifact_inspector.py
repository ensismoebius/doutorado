import json

import numpy as np
import pytest

pytest.importorskip("PySide6")

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.artifact_inspector import ArtifactInspector, _report


def test_report_csv(tmp_path):
    p = tmp_path / "x_paraconsistent.csv"
    p.write_text("label,alpha,d_penalized\na,0.5,1.2\nb,0.7,0.9\n")
    r = _report(p)
    assert "2 rows x 3 columns" in r
    assert "alpha: min=0.5 max=0.7" in r
    assert "Size" in r and "Format : csv" in r


def test_report_json_lists_keys(tmp_path):
    p = tmp_path / "s_summary.json"
    p.write_text(json.dumps({"modality": "eeg", "seed": 42}))
    r = _report(p)
    assert "Top-level keys: ['modality', 'seed']" in r


def test_report_npy_stats_and_nan(tmp_path):
    a = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
    p = tmp_path / "arr.npy"
    np.save(p, a)
    r = _report(p)
    assert "shape = (2, 2)" in r and "dtype = float32" in r
    assert "NaN   = 1" in r and "Inf   = 0" in r


def test_report_jsonl_first_last(tmp_path):
    p = tmp_path / "e_events.jsonl"
    p.write_text('{"type":"session_begin"}\n{"type":"epoch","epoch":1}\n{"type":"session_end"}\n')
    r = _report(p)
    assert "Records: 3" in r and "session_begin" in r and "session_end" in r


class _Adapter:
    def __init__(self, files):
        self._files = files

    def artifact_files(self, node):
        return self._files


def test_view_no_artifact_message(qapp):
    repo = DataRepository()
    repo._adapters["thesis"] = _Adapter([])
    v = ArtifactInspector(repo)
    v.show_node(TreeNode("sample", "s", {}), "thesis")
    assert "recomputed live" in v._text.toPlainText()


def test_view_lists_and_renders(qapp, tmp_path):
    f = tmp_path / "r_metrics.csv"
    f.write_text("eer,auc\n0.1,0.95\n")
    repo = DataRepository()
    repo._adapters["thesis"] = _Adapter([str(f)])
    v = ArtifactInspector(repo)
    v.show_node(TreeNode("run", "r", {}), "thesis")
    assert v._picker.count() == 1
    assert "1 rows x 2 columns" in v._text.toPlainText()
