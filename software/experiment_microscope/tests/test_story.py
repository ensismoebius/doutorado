"""Guided-tour engine + semantic palette + verdict sentences."""

import pytest

pytest.importorskip("PySide6")

from experiment_microscope.core import palette, verdict
from experiment_microscope.core.story import STORIES


def test_palette_is_consistent():
    for name in ("input", "output", "error", "spike", "membrane", "threshold"):
        assert palette.rgb(name) != (200, 200, 200)          # defined
        assert palette.hex_(name).startswith("#") and len(palette.hex_(name)) == 7
        assert name in palette.MEANING and name in palette.MEANING_LONG


def test_verdict_sentences_are_plain_and_safe():
    from experiment_microscope.core.integrity import assert_no_inferential_language

    for s in (verdict.reconstruction(0.99), verdict.reconstruction(0.4),
              verdict.reconstruction(None), verdict.spikes(60, 40, 256),
              verdict.paraconsistent(0.26, 0.94), verdict.latent(32)):
        assert isinstance(s, str) and len(s) > 10
        assert_no_inferential_language(s)   # no "best"/"optimal"/"significant"


def test_stories_well_formed():
    for key in ("meeting01", "thesis"):
        st = STORIES[key]
        assert len(st.steps) >= 5
        assert st.steps[-1].tab == ""           # last step = free explore, no view
        for s in st.steps[:-1]:
            assert s.narration and s.tab


def test_story_panel_drives_the_workspace(qapp):
    import experiment_microscope.app.workspace as W

    w = W.Workspace()
    w._start_tour()
    sp = w._story_panel
    assert sp._story.key == "meeting01"
    seen = set()
    for _ in range(len(sp._story.steps)):
        seen.add(w.tabs.tabText(w.tabs.currentIndex()))
        sp._advance()
    # the tour visited several different tabs
    assert len(seen) >= 3
    # finishing restores every tab
    assert all(w.tabs.isTabVisible(i) for i in range(w.tabs.count()))
