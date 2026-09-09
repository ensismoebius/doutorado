import pytest

pytest.importorskip("PySide6")

from experiment_microscope.data.repository import DataRepository
from experiment_microscope.views.explorer import ExplorerTree
from experiment_microscope.views.search_bar import SearchBar


def test_catalog_covers_experiments(qapp):
    tree = ExplorerTree(DataRepository())
    cat = tree.build_catalog(max_depth=2)
    tops = {e["path"][0] for e in cat}
    assert {"Meeting01", "Thesis"} <= tops


def test_search_tokens_all_must_match(qapp):
    tree = ExplorerTree(DataRepository())
    bar = SearchBar(tree)
    bar._edit.setText("thesis")
    n_thesis = bar._results.count()
    assert n_thesis >= 1
    bar._edit.setText("thesis zzznotarealtoken")
    assert bar._results.count() == 0


def test_activating_a_hit_emits_its_path(qapp):
    tree = ExplorerTree(DataRepository())
    bar = SearchBar(tree)
    seen = []
    bar.path_activated.connect(seen.append)
    bar._edit.setText("thesis")
    bar._activate_item(bar._results.item(0))
    assert seen and seen[0][0] == "Thesis"


def test_empty_query_hides_results(qapp):
    tree = ExplorerTree(DataRepository())
    bar = SearchBar(tree)
    bar._edit.setText("meeting01")
    assert bar._results.isVisibleTo(bar) and bar._results.count() >= 1
    bar._edit.setText("")
    assert not bar._results.isVisibleTo(bar) and bar._results.count() == 0
