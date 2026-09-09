import pytest

pytest.importorskip("PySide6")

from experiment_microscope.core.integrity import assert_no_inferential_language
from experiment_microscope.views.comparison_view import (
    DIFFERENT,
    NA,
    SAME,
    SIMILAR,
    UNKNOWN,
    ComparisonView,
    _ROWS,
)


def test_every_row_has_a_known_relation():
    valid = {SAME, SIMILAR, DIFFERENT, NA, UNKNOWN}
    assert all(r.relation in valid for r in _ROWS)


def test_no_inferential_language_in_cells():
    for r in _ROWS:
        for text in (r.aspect, r.meeting01, r.thesis):
            assert_no_inferential_language(text, context=r.aspect)


def test_does_not_claim_equivalence():
    # the strongest relation offered is "same concept" — never "equivalent" / "identical"
    joined = " ".join(r.relation for r in _ROWS).lower()
    assert "equivalent" not in joined and "identical" not in joined


def test_table_renders_all_rows(qapp):
    v = ComparisonView()
    assert v._table.rowCount() == len(_ROWS)
    assert v._table.item(0, 3).text() in {SAME, SIMILAR, DIFFERENT, NA, UNKNOWN}
    counts = v.relation_counts()
    assert sum(counts.values()) == len(_ROWS)


def test_show_node_is_noop(qapp):
    v = ComparisonView()
    v.show_node(object(), "meeting01")  # must not raise
