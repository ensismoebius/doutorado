"""Adapter discovery + graceful degradation (FIXME §49).

These run against the real ``software/nn/results/`` tree. thesis + paraconsistentGA
artifacts exist today; meeting01 results were purged, so the meeting01 adapter
must still populate its tree from the profile and never raise.
"""

import pytest

from experiment_microscope.data.adapters import TreeNode
from experiment_microscope.data.meeting01_adapter import Meeting01Adapter
from experiment_microscope.data.paraconsistent_ga_adapter import ParaconsistentGaAdapter
from experiment_microscope.data.repository import DataRepository
from experiment_microscope.data.thesis_adapter import ThesisAdapter


def test_meeting01_tree_without_results():
    a = Meeting01Adapter()
    roots = a.root_nodes()
    assert roots and roots[0].kind == "experiment"
    datasets = a.children(roots[0])
    # profile lists fsdd / audiomnist / mitbih even with zero runs
    assert {n.label for n in datasets} >= {"fsdd"}
    folds = a.children(datasets[0])
    assert len(folds) >= 1
    children = a.children(folds[0])
    # A fold's own children are windows groups only now — models never sit as
    # dead-end leaves here (selecting one used to render nothing, since a
    # model spec alone carries no window). They live under each WINDOW
    # instead, where selecting one actually runs against real data (FIXME §55).
    assert children and all(n.handle.get("level") == "windows" for n in children)
    windows_group = next((n for n in children if n.handle.get("split") == "test"), children[0])
    wins = a.children(windows_group)
    if not wins:
        return  # no split manifest / FSDD corpus in this environment
    models = a.children(wins[0])
    assert all(n.kind == "model" for n in models)
    if models:
        # provenance on a per-window model leaf must not raise even with no
        # events file for it
        prov = a.load_provenance(models[0])
        assert "run_tag" in prov.artifact


def test_compare_models_needs_a_window():
    a = Meeting01Adapter()
    with pytest.raises(NotImplementedError):
        a.compare_models(TreeNode("x", "root", {"level": "root"}))


def test_compare_models_only_this_exact_fold(first_fsdd_window):
    node, a = first_fsdd_window()
    if node is None or not a._snn_model_specs():
        pytest.skip("no FSDD corpus or trained model")
    rows = a.compare_models(node)
    assert rows  # this fold has at least one trained model
    fold = node.handle["cv_fold"]
    assert all(r["spec"]["fold"] == fold for r in rows)
    ok = [r for r in rows if "metrics" in r]
    assert ok and all({"mse", "mae", "r2"} <= r["metrics"].keys() for r in ok)


def test_thesis_paraconsistent_points_have_six_quantities():
    a = ThesisAdapter()
    points = a.paraconsistent_points()
    if not points:
        return  # no thesis results checked out
    p = points[0]
    for field in (p.alpha, p.beta, p.g1, p.g2, p.d_truth, p.d_penalized):
        assert hasattr(field, "origin")


def test_paraconsistent_ga_adapter_tolerates_absence():
    a = ParaconsistentGaAdapter()
    roots = a.root_nodes()
    assert roots
    # children must not raise regardless of whether results exist
    a.children(roots[0])


def test_repository_aggregates_points():
    repo = DataRepository()
    pts = repo.all_paraconsistent_points()
    assert isinstance(pts, list)
