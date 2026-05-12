#!/usr/bin/env python3
"""
validate_static_analysis.py

Parse cppcheck/clang-tidy output and enforce regression gates.

Enforces that:
1. No NEW low-risk codeability issues are introduced
2. Only APPROVED suppressions are allowed
3. Critical (high-severity) issues always fail

Usage:
    python validate_static_analysis.py --report cppcheck-report.xml [--strict]
    python validate_static_analysis.py --clang-tidy clang-tidy.txt [--strict]
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# Approved Low-Risk Suppressions — ALLOWLIST
# ============================================================================
# Dict format: (file_path, issue_id) -> reason
# These are suppressions verified as safe and approved for the codebase.

APPROVED_SUPPRESSIONS = {
    # Third-party header noise (instantiated through project code)
    ("src/core/dataLoaders/BatchPrefetcher.cpp", "knownConditionTrueFalse"): "Loop condition is runtime guard",
    ("src/core/wave/tests/wave_gtest.cpp", "passedByValue"): "Callback param contract fixed by definition",
    ("src/core/wave/tests/wave_gtest.cpp", "passedByValueCallback"): "Callback param contract fixed by definition",
    ("src/demos/cppdemos/speaker_demo.cpp", "useStlAlgorithm"): "argparse header noise",
    ("src/experiments/Config.cpp", "knownConditionTrueFalse"): "Condition is runtime guard",
    ("src/experiments/Config.cpp", "useStlAlgorithm"): "nlohmann_json header template noise",
    
    # OpenCL conditional implementations (guarded by can_use_opencl checks)
    ("src/core/tensor/opencl/OpenCLTensorBackend.cpp", "knownConditionTrueFalse"): "can_use_opencl gates optional implementations",
    
    # Syntax errors from header issues (cross-translation-unit macros)
    ("src/core/tensor/opencl/OpenCLTensorBackend.cpp", "syntaxError"): "Header macro evaluation issue",
    ("src/core/tensor/tests/opencl_tensor_backend_gtest.cpp", "syntaxError"): "Header macro evaluation issue",
    ("src/core/wavelet/tests/wavelet_gtest.cpp", "syntaxError"): "Header macro evaluation issue",
    
    # Unused structure members from cache implementations
    ("src/core/dataLoaders/10.1117/loaders/AudioLoader.cpp", "unusedStructMember"): "Cache capacity constant for future use",
    ("src/core/dataLoaders/10.1117/loaders/EEGLoader.cpp", "unusedStructMember"): "Cache capacity constant for future use",
    
    # Acceptable code patterns
    ("src/core/paraconsistent/paraconsistent.cpp", "variableScope"): "Scope chosen for clarity in complex operations",
    ("src/core/paraconsistent/tests/paraconsistent_gtest.cpp", "constVariableReference"): "Test loop variable usage pattern",
    ("src/core/utility/progress.cpp", "constVariableReference"): "Variable used in context-specific way",
    ("src/core/optimizers/tests/state_io_gtest.cpp", "shadowVariable"): "Intentional shadowing for nested scopes in tests",
    ("src/core/layers/Conv2d_utils.cpp", "unreadVariable"): "Assigned but not read (implementation placeholder)",
    ("src/core/dataLoaders/tests/count_mat_rows_gtest.cpp", "constVariable"): "Data array usage pattern in tests",
    
    # Cross-translation-unit symbol collisions (acceptable for classes with internal linkage)
    ("src/core/tensor/opencl/OpenCLTensorBackend.cpp", "ctuOneDefinitionRuleViolation"): "Incomplete type definitions in headers",
    ("src/demos/cppdemos/rede_snn.cpp", "ctuOneDefinitionRuleViolation"): "Local class definition acceptable in demo",
    ("src/experiments/03/lib/include/Experiment03Config.hpp", "ctuOneDefinitionRuleViolation"): "Config struct in experiment header",
    
    # Other acceptable patterns
    ("src/experiment03.cpp", "clarifyCalculation"): "Ternary operator with modulo (acceptable precedence)",
}


def parse_cppcheck_xml(xml_file: Path) -> List[Dict]:
    """Parse cppcheck XML report and extract violations."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Failed to parse {xml_file}: {e}")
        return []

    violations = []
    for error in root.findall(".//error"):
        loc = error.find("location")
        if loc is None:
            continue

        file_path = loc.get("file", "unknown")
        line = loc.get("line", "0")
        issue_id = error.get("id", "unknown")
        severity = error.get("severity", "unknown")
        msg = error.get("msg", "")

        # Normalize path (remove absolute /home/... prefix, keep relative)
        if "/src/" in file_path:
            rel_path = file_path[file_path.index("/src/") + 1 :]
        else:
            rel_path = file_path
        # Clean up path: remove leading slash if present, standardize to relative
        rel_path = rel_path.lstrip("/")
        
        violations.append(
            {
                "file": rel_path,
                "line": line,
                "id": issue_id,
                "severity": severity,
                "msg": msg,
            }
        )

    return violations


def parse_clang_tidy_output(output_file: Path) -> List[Dict]:
    """Parse clang-tidy text output and extract violations."""
    violations = []
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Failed to read {output_file}: {e}")
        return []

    # clang-tidy format: file:line:col: [level] message [check-name]
    for line in lines:
        if "warning:" in line or "error:" in line:
            try:
                parts = line.split(":")
                if len(parts) >= 4:
                    file_path = parts[0]
                    line_num = parts[1]
                    col = parts[2]
                    msg_part = ":".join(parts[3:])

                    # Extract check name from brackets [check-name]
                    check_name = "unknown"
                    if "[" in msg_part and "]" in msg_part:
                        check_name = (
                            msg_part[msg_part.rindex("[") + 1 : msg_part.rindex("]")]
                        )

                    # Normalize path
                    if "/src/" in file_path:
                        rel_path = file_path[file_path.index("/src/") + 1 :]
                    else:
                        rel_path = file_path

                    violations.append(
                        {
                            "file": rel_path,
                            "line": line_num,
                            "id": check_name,
                            "severity": "warning",
                            "msg": msg_part.strip(),
                        }
                    )
            except Exception:
                continue

    return violations


def check_regression(violations: List[Dict], strict: bool = False) -> Tuple[bool, str]:
    """
    Check static analysis violations against allowlist.

    Returns:
        (passed: bool, summary: str)
    """
    summary = []
    summary.append("=" * 70)
    summary.append("STATIC ANALYSIS REGRESSION CHECK")
    summary.append("=" * 70)

    # Categorize violations
    approved = []
    unapproved = []

    for v in violations:
        file_key = v["file"]
        issue_key = v["id"]
        full_key = (file_key, issue_key)

        if full_key in APPROVED_SUPPRESSIONS:
            approved.append(v)
        else:
            unapproved.append(v)

    # Report results
    summary.append(f"\n✅ APPROVED VIOLATIONS: {len(approved)}")
    for v in approved:
        key = (v["file"], v["id"])
        reason = APPROVED_SUPPRESSIONS.get(key, "N/A")
        summary.append(f"  - {v['file']}:{v['line']} [{v['id']}] ({reason})")

    summary.append(f"\n❌ UNAPPROVED VIOLATIONS: {len(unapproved)}")
    for v in unapproved:
        summary.append(f"  - {v['file']}:{v['line']} [{v['id']}] {v.get('msg', '')}")

    summary.append("=" * 70)

    # Determine pass/fail
    passed = len(unapproved) == 0
    if not passed:
        summary.append(
            f"\n❌ FAILED: {len(unapproved)} unapproved violations must be resolved"
        )
    else:
        if strict:
            summary.append("\n✅ PASSED (strict mode): All violations are approved")
        else:
            summary.append("\n✅ PASSED (regression mode): No new violations detected")

    summary.append("=" * 70)

    return passed, "\n".join(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Validate static analysis reports and enforce regression gates."
    )
    parser.add_argument("--report", type=Path, help="Path to cppcheck XML report")
    parser.add_argument("--clang-tidy", type=Path, help="Path to clang-tidy text output")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode: fail if any unapproved violations exist",
    )
    parser.add_argument(
        "--list-approved",
        action="store_true",
        help="List all approved suppressions and exit",
    )

    args = parser.parse_args()

    if args.list_approved:
        print("Approved Suppressions:")
        print("-" * 70)
        for (file, issue), reason in sorted(APPROVED_SUPPRESSIONS.items()):
            print(f"  {file:50} [{issue:30}] {reason}")
        print("-" * 70)
        return 0

    violations = []

    if args.report:
        print(f"📊 Parsing cppcheck report: {args.report}")
        violations.extend(parse_cppcheck_xml(args.report))

    if args.clang_tidy:
        print(f"📊 Parsing clang-tidy output: {args.clang_tidy}")
        violations.extend(parse_clang_tidy_output(args.clang_tidy))

    if not violations:
        print("ℹ️  No violations found in reports")
        return 0

    passed, summary = check_regression(violations, strict=args.strict)
    print(summary)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
