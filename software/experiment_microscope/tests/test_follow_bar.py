from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.follow_data import STAGES, FollowDataBar


def _enabled(bar) -> set[str]:
    return {s.key for s in STAGES if bar._buttons[s.key].isEnabled()}


def test_no_selection_disables_every_stage(qapp):
    bar = FollowDataBar(DataRepository())
    bar.update_for(None, None)
    assert _enabled(bar) == set()


def test_meeting01_window_lights_signal_and_wavelet(qapp):
    bar = FollowDataBar(DataRepository())
    node = TreeNode("sample", "win 0", {"level": "window", "dataset": "fsdd", "cv_fold": 0,
                                        "split": "test", "row": 0})
    bar.update_for(node, "meeting01")
    en = _enabled(bar)
    assert {"raw", "window", "normalized", "wavelet", "paraconsistent"} <= en
    assert "latent" not in en  # no trained model


def test_thesis_run_lights_features(qapp):
    bar = FollowDataBar(DataRepository())
    node = TreeNode("run", "r", {"level": "run", "phase": "phase00", "run_tag": "x"})
    bar.update_for(node, "thesis")
    assert "features" in _enabled(bar)


def test_stage_click_emits_tab_name(qapp):
    bar = FollowDataBar(DataRepository())
    seen = []
    bar.stage_activated.connect(seen.append)
    node = TreeNode("sample", "s", {"level": "window"})
    bar.update_for(node, "meeting01")
    bar._buttons["wavelet"].click()
    assert seen == ["Wavelet Lab"]
