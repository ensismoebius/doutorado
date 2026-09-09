import math

import pytest

from experiment_microscope.core.integrity import (
    InferentialLanguageError,
    Origin,
    Value,
    assert_no_inferential_language,
)


def test_missing_never_renders_zero():
    v = Value.missing()
    assert v.is_missing
    assert v.display() == "—"
    assert v.labelled() == "—"


def test_nan_magnitude_is_missing():
    v = Value(float("nan"), Origin.MEASURED)
    assert v.is_missing
    assert v.display() == "—"


def test_measured_value_carries_origin_tag():
    v = Value(0.1583, Origin.COMPUTED, unit="")
    assert "computed" in v.labelled()
    assert v.display().startswith("0.1583")


def test_banned_word_rejected():
    with pytest.raises(InferentialLanguageError):
        assert_no_inferential_language("this run converged after 12 epochs")


def test_substring_not_flagged():
    # "convergence" contains "converge"? guard is word-boundary; "coverage" must pass
    assert assert_no_inferential_language("channel coverage is 6/6")


def test_neutral_caption_passes():
    text = "validation MSE 0.0142 at epoch 9; training MSE 0.0091"
    assert assert_no_inferential_language(text) == text
