import pytest

pytest.importorskip("PySide6")

from experiment_microscope.core import glossary as g


def test_core_terms_present_and_described():
    for term in ("MSE", "MAE", "R2", "SNN", "LIF", "LOSO", "PCA", "t-SNE",
                 "D_penalized", "G1", "G2", "v_th", "ZCR", "EER", "AUC", "latent",
                 "z-score", "NSGA-II", "UNCALIBRATED"):
        assert g.describe(term), f"{term} has no plain-language description"


def test_expand_and_aliases():
    assert g.expand("MSE") == "MSE (Mean Squared Error)"
    assert "correlation" in g.expand("pearson_r").lower()
    assert g.describe("r²") == g.describe("R2")      # alias
    assert g.expand("totally-unknown-term") == "totally-unknown-term"


def test_tooltip_and_html():
    tip = g.tooltip("EER")
    assert "Equal Error Rate" in tip and "—" in tip
    html = g.glossary_html()
    assert html.startswith("<h3>") and "Spiking Neural Network" in html


def test_helpbox_toggles(qapp):
    from experiment_microscope.views._help import HelpBox

    box = HelpBox("Test", "<b>hello</b> world")
    assert not box._body.isVisibleTo(box)
    box.open()
    assert box._body.isVisibleTo(box)


def test_every_central_tab_has_a_helpbox(qapp):
    from PySide6.QtWidgets import QWidget

    import experiment_microscope.app.workspace as W

    w = W.Workspace()
    missing = []
    for i in range(w.tabs.count()):
        view = w.tabs.widget(i)
        has = any(c.__class__.__name__ == "HelpBox" for c in view.findChildren(QWidget))
        if not has:
            missing.append(w.tabs.tabText(i))
    # Pipeline is a bare DAG diagram; everything else must guide the reader.
    assert missing in ([], ["Pipeline"]), f"tabs with no explanation: {missing}"
