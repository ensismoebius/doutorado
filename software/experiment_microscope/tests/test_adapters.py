"""Adapter discovery + graceful degradation (FIXME §49).

These run against the real ``software/nn/results/`` tree. thesis + paraconsistentGA
artifacts exist today; meeting01 results were purged, so the meeting01 adapter
must still populate its tree from the profile and never raise.
"""

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
    combos = [n for n in children if n.kind == "model"]
    assert combos, "fold should expose snn-ae recompute leaves"
    assert any(n.handle.get("level") == "windows" for n in children)
    # provenance on a combo leaf must not raise even if no events file exists
    prov = a.load_provenance(combos[0])
    assert "run_tag" in prov.artifact


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
