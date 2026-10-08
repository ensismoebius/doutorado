"""ControlsWidget's parameter sliders: hidden by default, shown only on
request, and only for a demo that actually has parameters at all.

Before this file existed, a demo's sliders appeared the instant it had any
-- every viewer saw every parameter control for every such demo, whether
they wanted to tweak one or not. "Mostrar parâmetros" (mirroring "Mostrar
equação"/"Mostrar explicação", both already opt-in) makes that opt-in too.
"""

from efficient_nn_lab.widgets.controls import ControlsWidget

_ONE_PARAM = {"amplitude": {"label": "Amplitude", "min": 0.0, "max": 1.0, "step": 0.05, "value": 0.3}}


def test_params_button_and_sliders_start_hidden(qapp):
    controls = ControlsWidget()
    controls.show()
    assert not controls._params_btn.isChecked()
    assert not controls._params_container.isVisible()


def test_params_button_appears_only_for_a_demo_with_parameters(qapp):
    controls = ControlsWidget()
    controls.show()
    assert not controls._params_btn.isVisible()

    controls.rebuild_parameters(_ONE_PARAM)
    assert controls._params_btn.isVisible()

    controls.rebuild_parameters({})
    assert not controls._params_btn.isVisible()


def test_checking_the_button_reveals_the_sliders(qapp):
    controls = ControlsWidget()
    controls.show()
    controls.rebuild_parameters(_ONE_PARAM)
    assert not controls._params_container.isVisible()

    controls._params_btn.setChecked(True)
    assert controls._params_container.isVisible()


def test_switching_to_a_demo_without_parameters_hides_an_open_panel(qapp):
    """Regression: rebuilding with an empty spec used to leave the
    container exactly as visible as it was -- so a panel toggled open on
    one demo stayed visible, now empty, after switching to a demo with no
    parameters at all.
    """
    controls = ControlsWidget()
    controls.show()
    controls.rebuild_parameters(_ONE_PARAM)
    controls._params_btn.setChecked(True)
    assert controls._params_container.isVisible()

    controls.rebuild_parameters({})
    assert not controls._params_btn.isChecked()
    assert not controls._params_container.isVisible()


def test_toggle_stays_open_across_demos_that_both_have_parameters(qapp):
    """The toggle is a standing preference, like "Mostrar equação" --
    rebuilding for the NEXT demo must not silently close a panel the
    viewer deliberately opened, as long as that next demo also has
    something to show in it.
    """
    controls = ControlsWidget()
    controls.show()
    controls.rebuild_parameters(_ONE_PARAM)
    controls._params_btn.setChecked(True)

    other_param = {"tau": {"label": "tau", "min": 1.0, "max": 15.0, "step": 0.5, "value": 5.0}}
    controls.rebuild_parameters(other_param)
    assert controls._params_btn.isChecked()
    assert controls._params_container.isVisible()
