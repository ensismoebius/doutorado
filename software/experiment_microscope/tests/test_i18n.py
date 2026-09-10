"""Runtime translation + busy indicator + scale-to-fit helper."""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from experiment_microscope.core import i18n, verdict


def test_translation_falls_back_to_english():
    i18n.set_language("en")
    assert i18n.t("Free explore") == "Free explore"
    assert i18n.t("a string with no catalog entry ever") == "a string with no catalog entry ever"
    assert i18n.t("{n} apples", n=3) == "3 apples"


def test_pt_br_translates_the_didactic_surface():
    i18n.set_language("pt_BR")
    try:
        assert i18n.t("Free explore") == "Explorar livremente"
        assert i18n.t("Start guided tour") != ""            # present or English, never blank
        s = verdict.reconstruction(0.99)
        assert "quase" in s.lower() and "R²" in s           # translated + keeps the symbol
        sp = verdict.spikes(60, 40, 256)
        assert "disparos" in sp
    finally:
        i18n.set_language("en")


def test_language_choice_persists_round_trip():
    i18n.set_language("pt_BR")
    assert i18n.language() == "pt_BR"
    i18n.set_language("nonsense")
    assert i18n.language() == "en"


def test_busy_indicator_refcounts(qapp):
    from experiment_microscope.views._busy import BusyIndicator

    b = BusyIndicator()
    assert not b.isVisible()
    b.begin("Working…")
    b.begin("Loading data…")
    assert b.isVisible()
    b.end()
    assert b.isVisible()          # one task still open
    b.end()
    assert not b.isVisible()


def test_autofit_is_safe_without_data(qapp):
    import pyqtgraph as pg

    from experiment_microscope.views._plotinfo import autofit, fade_in

    w = pg.PlotWidget()
    autofit(w)                    # empty plot — must not raise
    w.plot(np.arange(5), np.arange(5) * 3.0)
    autofit(w)
    fade_in(w)
