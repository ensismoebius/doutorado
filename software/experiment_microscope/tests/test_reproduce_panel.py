import json

import pytest

pytest.importorskip("PySide6")

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.reproduce_panel import ReproducePanel


class _Thesis:
    def __init__(self, summary_path):
        self._sp = str(summary_path)

    def artifact_files(self, node):
        return [self._sp]


class _Meeting01:
    def artifact_files(self, node):
        return []

    def load_provenance(self, node):
        from experiment_microscope.data.adapters import ProvenanceRecord
        from experiment_microscope.core.integrity import Origin, Value
        v = Value("fsdd", Origin.MEASURED)
        return ProvenanceRecord(source={"dataset": v}, processing={}, model={}, artifact={})


def _repo(**adapters):
    r = DataRepository()
    r._adapters.update(adapters)
    return r


def test_thesis_recipe_names_phase_script_and_params(qapp, tmp_path):
    sp = tmp_path / "e05_p00_hc_daub10_lfcc_c1_eeg_rep0_summary.json"
    sp.write_text(json.dumps({"seed": 42, "modality": "eeg", "config_hash": 123, "strategy": "handcrafted"}))
    v = ReproducePanel(_repo(thesis=_Thesis(sp)))
    v.show_node(TreeNode("run", "r", {"level": "run", "phase": "phase00",
                                      "run_tag": "e05_p00_hc_daub10_lfcc_c1_eeg_rep0"}), "thesis")
    cmd = v._cmd.toPlainText()
    assert "run_thesis_profiles.sh phase00" in cmd
    assert "seed=42" in cmd and "config_hash=123" in cmd


def test_meeting01_recipe_is_guarded_and_per_fold(qapp):
    v = ReproducePanel(_repo(meeting01=_Meeting01()))
    v.show_node(TreeNode("sample", "w", {"level": "window", "dataset": "fsdd", "cv_fold": 2}), "meeting01")
    cmd = v._cmd.toPlainText()
    assert "EXPERIMENT_CONFIRMED=1" in cmd
    assert "--dataset fsdd --cv-fold 2" in cmd


def test_non_run_node_clears_recipe(qapp):
    v = ReproducePanel(_repo(thesis=_Thesis("/nope")))
    v.show_node(TreeNode("experiment", "Thesis", {"level": "root"}), "thesis")
    assert v._cmd.toPlainText() == ""
    assert not v._copy_btn.isEnabled()


def test_copy_command_to_clipboard(qapp):
    v = ReproducePanel(_repo(meeting01=_Meeting01()))
    v.show_node(TreeNode("sample", "w", {"level": "window", "dataset": "mitbih", "cv_fold": 0}), "meeting01")
    v._copy_cmd()
    from PySide6.QtWidgets import QApplication
    assert "--dataset mitbih" in QApplication.clipboard().text()
