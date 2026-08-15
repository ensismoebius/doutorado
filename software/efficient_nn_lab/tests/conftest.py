"""Headless Qt setup shared by the GUI smoke tests.

Forces the offscreen platform plugin before PySide6 is imported anywhere,
so the test suite runs without a display (CI, this sandbox, etc.).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
