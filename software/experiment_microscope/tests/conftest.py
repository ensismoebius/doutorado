"""Headless Qt setup shared by the GUI smoke tests.

Forces the offscreen platform plugin before PySide6 is imported anywhere, so
the suite runs without a display (CI, sandbox).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def first_fsdd_window():
    """The first FSDD test-fold window TreeNode, or None if the corpus /
    binding is unavailable. Shared by the follow-the-data tests."""

    def _get():
        from experiment_microscope.data.meeting01_adapter import Meeting01Adapter

        a = Meeting01Adapter()
        root = a.root_nodes()[0]
        try:
            fsdd = next(n for n in a.children(root) if n.label == "fsdd")
            fold0 = a.children(fsdd)[0]
            group = next(n for n in a.children(fold0) if n.handle.get("level") == "windows")
            wins = a.children(group)
        except (StopIteration, RuntimeError):
            return None, a
        return (wins[0] if wins else None), a

    return _get
