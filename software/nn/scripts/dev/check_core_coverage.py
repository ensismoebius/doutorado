#!/usr/bin/env python3
"""Fail the build unless core library coverage reaches required thresholds.

This script captures lcov data from an instrumented build tree, filters to core
library files, computes per-file line/function coverage, and exits non-zero when
any file is below thresholds.

Default policy: enforce 100% function/method coverage for core files.
Line coverage is optional and only enforced when ``--line-threshold`` is > 0.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class FileCoverage:
    line_total: int = 0
    line_hit: int = 0
    fn_total: int = 0
    fn_hit: int = 0

    @property
    def line_rate(self) -> float:
        if self.line_total == 0:
            return 100.0
        return (100.0 * self.line_hit) / self.line_total

    @property
    def fn_rate(self) -> float:
        if self.fn_total == 0:
            return 100.0
        return (100.0 * self.fn_hit) / self.fn_total


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _get_excluded_lines(source_file: Path) -> set[int]:
    """Return set of 1-based line numbers annotated with LCOV_EXCL_LINE."""
    excluded: set[int] = set()
    try:
        for i, src_line in enumerate(
            source_file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if "LCOV_EXCL_LINE" in src_line:
                excluded.add(i)
    except OSError:
        pass
    return excluded


def apply_source_exclusions(info_path: Path) -> None:
    """Rewrite info file removing DA entries annotated  in source."""
    content = info_path.read_text(encoding="utf-8", errors="replace")

    out_sections: list[str] = []
    for section in content.split("end_of_record"):
        if not section.strip():
            out_sections.append(section)
            continue

        # Find source file path
        source_file: Path | None = None
        for raw_line in section.splitlines():
            if raw_line.startswith("SF:"):
                source_file = Path(raw_line[3:].strip())
                break

        if source_file is None or not source_file.exists():
            out_sections.append(section + "end_of_record")
            continue

        excluded = _get_excluded_lines(source_file)
        if not excluded:
            out_sections.append(section + "end_of_record")
            continue

        # Rebuild section without excluded DA entries
        kept_da: list[tuple[int, int]] = []
        new_lines: list[str] = []
        for raw_line in section.splitlines(keepends=True):
            stripped = raw_line.strip()
            if stripped.startswith("DA:"):
                parts = stripped[3:].split(",")
                lineno = int(parts[0])
                if lineno in excluded:
                    continue
                kept_da.append((lineno, int(parts[1])))
            elif stripped.startswith("LH:") or stripped.startswith("LF:"):
                continue  # Will be recalculated
            new_lines.append(raw_line)

        lf = len(kept_da)
        lh = sum(1 for _, hits in kept_da if hits > 0)

        # Insert recalculated LH/LF after the FNF line
        final_lines: list[str] = []
        inserted = False
        for raw_line in new_lines:
            final_lines.append(raw_line)
            if not inserted and raw_line.strip().startswith("FNF:"):
                final_lines.append(f"LF:{lf}\n")
                final_lines.append(f"LH:{lh}\n")
                inserted = True

        if not inserted:
            final_lines.append(f"LF:{lf}\n")
            final_lines.append(f"LH:{lh}\n")

        out_sections.append("".join(final_lines) + "end_of_record")

    info_path.write_text("".join(out_sections), encoding="utf-8")


def parse_lcov_info(info_path: Path) -> Dict[Path, FileCoverage]:
    data: Dict[Path, FileCoverage] = {}
    current: Path | None = None

    with info_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("SF:"):
                current = Path(line[3:]).resolve()
                data.setdefault(current, FileCoverage())
                continue

            if current is None:
                continue

            cov = data[current]
            if line.startswith("LH:"):
                cov.line_hit = int(line[3:])
            elif line.startswith("LF:"):
                cov.line_total = int(line[3:])
            elif line.startswith("FNH:"):
                cov.fn_hit = int(line[4:])
            elif line.startswith("FNF:"):
                cov.fn_total = int(line[4:])
            elif line == "end_of_record":
                current = None

    return data


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enforce core-library coverage (function coverage by default)"
    )
    parser.add_argument("--build-dir", required=True, help="CMake build directory")
    parser.add_argument(
        "--line-threshold",
        type=float,
        default=0.0,
        help="Required line coverage percentage (set > 0 to enforce)",
    )
    parser.add_argument(
        "--function-threshold",
        type=float,
        default=100.0,
        help="Required function coverage percentage",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run ctest before coverage capture",
    )
    return parser


def _run_instrumented_tests(build_dir: Path) -> None:
    """Reset stale gcov data and run the suite so fresh `.gcda` counters exist to capture."""
    # Reset stale gcov runtime data to prevent checksum mismatches and stale counts.
    for gcda in build_dir.rglob("*.gcda"):
        gcda.unlink(missing_ok=True)

    run([
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "-E",
        "(_NOT_BUILT$|SpikeCountLossTest\\.ForwardAndBackward$|BatchPrefetcherRamCapTest\\.OversizedBatchStillMakesProgress$)",
        "-j4",
    ])


def _capture_core_coverage(build_dir: Path, raw_info: Path, filtered_info: Path) -> None:
    """Capture lcov data from `build_dir` and filter it down to `src/core/` + `include/`.

    Anchor both patterns to the project source root.

    This used to be "*/include/nn/*", which matched ZERO files: there is no include/nn/
    directory in this project -- the public headers live directly under include/
    (include/tensor/, include/layers/, ...). Two consequences: the gate silently measured
    only src/core/ while claiming to also cover the public headers, and on lcov 2.x an
    --extract pattern that matches nothing is a hard error (exit 25), so the gate could
    not run at all.

    The replacement must stay anchored to source_root: a bare "*/include/*" also matches
    /usr/include/c++/... and _deps/*/include/..., pulling ~450 dependency/system files
    into the "core" measurement and drowning the real number.
    """
    run([
        "lcov",
        "--capture",
        "--directory",
        str(build_dir),
        "--output-file",
        str(raw_info),
        "--ignore-errors",
        "inconsistent,negative,gcov,range",
    ])

    source_root = Path(__file__).resolve().parents[2]
    run([
        "lcov",
        "--extract",
        str(raw_info),
        f"{source_root}/src/core/*",
        f"{source_root}/include/*",
        "--output-file",
        str(filtered_info),
        "--ignore-errors",
        "unused",
    ])

    run([
        "lcov",
        "--remove",
        str(filtered_info),
        "--ignore-errors",
        "unused",
        "*/src/core/*/tests/*",
        "*/src/core/tools/*",
        "*/_deps/*",
        "/usr/*",
        "--output-file",
        str(filtered_info),
    ])

    apply_source_exclusions(filtered_info)


def _find_failing_files(
    coverage: Dict[Path, FileCoverage], function_threshold: float, line_threshold: float, enforce_line: bool
) -> list[tuple[Path, FileCoverage]]:
    """Every file below `function_threshold`, or below `line_threshold` when that's enforced."""
    failing: list[tuple[Path, FileCoverage]] = []
    for path, cov in sorted(coverage.items(), key=lambda item: str(item[0])):
        fn_fail = cov.fn_rate < function_threshold
        line_fail = enforce_line and (cov.line_rate < line_threshold)
        if fn_fail or line_fail:
            failing.append((path, cov))
    return failing


def _print_report(
    coverage: Dict[Path, FileCoverage],
    failing: list[tuple[Path, FileCoverage]],
    line_threshold: float,
    enforce_line: bool,
    function_threshold: float,
) -> None:
    total = len(coverage)
    passing = total - len(failing)
    print("\nCore coverage summary")
    print(f"- Files evaluated: {total}")
    print(f"- Files passing thresholds: {passing}")
    print(f"- Line threshold: {line_threshold:.1f}%")
    print(f"- Line threshold enforced: {'yes' if enforce_line else 'no'}")
    print(f"- Function threshold: {function_threshold:.1f}%")

    if not failing:
        return
    print("\nFiles below threshold:")
    for path, cov in failing:
        print(
            f"- {path}: lines {cov.line_rate:.1f}% ({cov.line_hit}/{cov.line_total}), "
            f"functions {cov.fn_rate:.1f}% ({cov.fn_hit}/{cov.fn_total})"
        )


def main() -> int:
    args = _build_arg_parser().parse_args()

    build_dir = Path(args.build_dir).resolve()
    if not build_dir.exists():
        print(f"Build directory does not exist: {build_dir}", file=sys.stderr)
        return 2

    coverage_dir = build_dir / "coverage-gate"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    raw_info = coverage_dir / "core_raw.info"
    filtered_info = coverage_dir / "core_filtered.info"

    if not args.skip_tests:
        _run_instrumented_tests(build_dir)

    _capture_core_coverage(build_dir, raw_info, filtered_info)

    coverage = parse_lcov_info(filtered_info)
    if not coverage:
        print("No coverage records found for core library files.", file=sys.stderr)
        return 2

    enforce_line = args.line_threshold > 0.0
    failing = _find_failing_files(coverage, args.function_threshold, args.line_threshold, enforce_line)
    _print_report(coverage, failing, args.line_threshold, enforce_line, args.function_threshold)

    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
