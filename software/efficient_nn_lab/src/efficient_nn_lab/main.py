"""Entry point. Run with either:

    python -m efficient_nn_lab
    python src/efficient_nn_lab/main.py

`--demo <slug>` opens directly on a given demo instead of the welcome
screen -- see DemoModule.slug and app/main_window.py's
`_select_demo_by_slug`, and the lecture deck's clickable "open in
software" links (documentation/08-lectures/fronteiras-bitnets-redes-pulso/
presentation.md, phase 3).
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from efficient_nn_lab.app.main_window import MainWindow
    from efficient_nn_lab.core.math_utils import seed_everything

    parser = argparse.ArgumentParser(prog="efficient-nn-lab", add_help=True)
    parser.add_argument(
        "--demo",
        default=None,
        help="Open directly on this demo's slug (e.g. 'snn.lif') instead of the welcome screen.",
    )
    # parse_known_args so Qt's own flags (-style, -platform, ...) pass through untouched.
    args, qt_args = parser.parse_known_args(sys.argv[1:])

    seed_everything()
    app = QApplication([sys.argv[0], *qt_args])
    window = MainWindow(initial_demo_slug=args.demo)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
