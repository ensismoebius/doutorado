from experiment_microscope.core.selection import SelectionState


def test_setting_field_emits_once(qapp):
    sel = SelectionState()
    seen = []
    sel.changed.connect(seen.append)
    sel.set("dataset", "fsdd")
    assert seen == ["dataset"]


def test_coarse_change_clears_finer_fields(qapp):
    sel = SelectionState()
    sel.set("experiment", "thesis")
    sel.set("dataset", "eeg")
    sel.set("window", 3)
    seen = []
    sel.changed.connect(seen.append)
    sel.set("experiment", "meeting01")
    # window (finer than experiment) is cleared and also emits
    assert "experiment" in seen and "window" in seen
    assert sel.get("window") is None
    assert sel.get("dataset") is None


def test_no_emit_on_identical_value(qapp):
    sel = SelectionState()
    sel.set("dataset", "fsdd")
    seen = []
    sel.changed.connect(seen.append)
    sel.set("dataset", "fsdd")
    assert seen == []


def test_snapshot_restore_roundtrip(qapp):
    sel = SelectionState()
    sel.update(experiment="thesis", dataset="voice", window=7)
    snap = sel.snapshot()
    sel.set("experiment", "meeting01")
    sel.restore(snap)
    assert sel.get("experiment") == "thesis"
    assert sel.get("window") == 7
