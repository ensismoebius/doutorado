"""Entry point.

    python -m experiment_microscope
    ./run.sh

``--experiment {meeting01,thesis,paraconsistent_ga}`` pre-selects a pipeline.
``--tour {meeting01,thesis}`` opens straight into the guided tour.
``--tab <name>`` raises a named central tab on start (e.g. ``--tab "Wavelet Lab"``).
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
    parser.add_argument(
        "--tour",
        default=None,
        choices=["meeting01", "thesis"],
        help="Open straight into the guided tour for this pipeline.",
    )
    parser.add_argument(
        "--tab",
        default=None,
        help="Raise this central tab on start (English name, e.g. 'Wavelet Lab').",
    )
    # parse_known_args so Qt's own flags (-platform offscreen, ...) pass through.
    args, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Experiment Microscope")
    window = Workspace(
        initial_experiment=args.experiment,
        initial_tour=args.tour,
        initial_tab=args.tab,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
