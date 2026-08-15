"""Entry point. Run with either:

    python -m efficient_nn_lab
    python src/efficient_nn_lab/main.py
"""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from efficient_nn_lab.app.main_window import MainWindow
    from efficient_nn_lab.core.math_utils import seed_everything

    seed_everything()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
