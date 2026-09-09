"""Entry point.

    python -m experiment_microscope
    ./run.sh

``--experiment {meeting01,thesis,paraconsistent_ga}`` pre-selects a pipeline.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from experiment_microscope.app.workspace import Workspace

    parser = argparse.ArgumentParser(prog="experiment-microscope", add_help=True)
    parser.add_argument(
        "--experiment",
        default=None,
        choices=["meeting01", "thesis", "paraconsistent_ga"],
        help="Open with this pipeline pre-selected in the explorer.",
    )
    # parse_known_args so Qt's own flags (-platform offscreen, ...) pass through.
    args, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Experiment Microscope")
    window = Workspace(initial_experiment=args.experiment)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
