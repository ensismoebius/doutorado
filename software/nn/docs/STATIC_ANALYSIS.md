# Static Analysis & Code Quality

## Overview

This project uses a comprehensive static analysis pipeline integrated into CI/CD to maintain code quality and catch potential issues early:

- **Cppcheck** — Detects bugs, dead code, and coding style issues
- **Clang-Tidy** — Enforces modern C++ practices and catches performance issues
- **Flawfinder** — Identifies security-sensitive code patterns

## Cppcheck Integration

### How It Works

The project uses **cppcheck 2.20+** with `--enable=warning,style,performance,portability,information` to catch a wide range of issues.

### Known Suppressions (Approved)

The following suppressions are **approved** because they represent either:
1. **Third-party header noise** — Issues in templated headers (nlohmann_json, argparse, etc.)
2. **Valid runtime guards** — Code patterns that are correct but appear problematic to static analysis
3. **Acceptable tradeoffs** — Performance patterns deemed appropriate for the codebase

| File | Issue | Reason |
|------|-------|--------|
| `src/core/dataLoaders/BatchPrefetcher.cpp` | `knownConditionTrueFalse` | Loop condition is a valid runtime guard checking `stop_requested_` |
| `src/core/wave/tests/wave_gtest.cpp` | `passedByValue` | Callback parameter contract fixed by definition |
| `src/demos/cppdemos/speaker_demo.cpp` | `useStlAlgorithm` | Argparse header template instantiation (third-party noise) |
| `src/experiments/Config.cpp` | `knownConditionTrueFalse` | Condition is a valid runtime guard |
| `src/experiments/Config.cpp` | `useStlAlgorithm` | Nlohmann_json header template instantiation (third-party) |

**View approved suppressions:**
```bash
python3 scripts/validate_static_analysis.py --list-approved
```

### Adding New Suppressions

**IMPORTANT:** Only add suppressions after thorough investigation confirming the issue is:
1. A false-positive specific to cppcheck's analysis
2. Acceptable code according to our standards
3. Not a real defect

#### Process:

1. **Verify the issue is valid** — Reproduce locally:
   ```bash
   cd build
   ninja analysis-cppcheck
   grep 'id="issueId"' cppcheck-report.xml
   ```

2. **Choose suppression strategy:**

   **Option A: Inline suppression** (for specific lines)
   ```cpp
   // cppcheck-suppress knownConditionTrueFalse
   if (stop_requested_) {  // Valid runtime guard
       break;
   }
   ```

   **Option B: File-level suppression** (for header template noise)
   ```cpp
   // cppcheck-suppress-file useStlAlgorithm
   // Reason: nlohmann_json template instantiation
   
   #include "include/my_header.hpp"
   ```

3. **Add to approved list** — Update `scripts/validate_static_analysis.py`:
   ```python
   APPROVED_SUPPRESSIONS = {
       # ... existing entries ...
       ("src/path/to/file.cpp", "issueId", "Reason: ..."),
   }
   ```

4. **Validate** — Ensure cppcheck accepts the suppression:
   ```bash
   cd build && ninja analysis-cppcheck && \
   python3 ../scripts/validate_static_analysis.py --report cppcheck-report.xml
   ```

5. **Update CHANGELOG.md** — Document the suppression and rationale.

## Clang-Tidy Integration

### Configuration (`.clang-tidy`)

The project uses a curated set of checks:
- **Core safety:** `bugprone-*`, `clang-analyzer-*`
- **Modern C++:** `modernize-*`
- **Readability:** `readability-*` (excluding length/magic-numbers)
- **Performance:** `performance-*`
- **Guidelines:** `cppcoreguidelines-*`

### Enabling Clang-Tidy During Compilation

To run clang-tidy during compilation (enabled in CI, optional locally):
```bash
cmake -S . -B build -DNN_ENABLE_CLANG_TIDY=ON
ninja -j$(nproc)
```

### Local Clang-Tidy Analysis

Run clang-tidy standalone on all sources:
```bash
cd build
ninja analysis-clang-tidy
```

Or on specific files:
```bash
clang-tidy --config-file=.clang-tidy -p build src/core/myfile.cpp
```

### Common Clang-Tidy Issues & Fixes

| Issue | Common Cause | Fix |
|-------|--------------|-----|
| `modernize-use-auto` | Type explicitly stated | Use auto where intent is clear |
| `readability-use-anyof` | Manual loops | Use STL algorithms (with caution) |
| `performance-unnecessary-value-param` | Copy-by-value | Use const-ref for large types |
| `bugprone-suspicious-string-compare` | String comparison edge case | Use `.compare() == 0` or `==` carefully |

## Flawfinder Integration

**Flawfinder** identifies security-sensitive code patterns (CWE/CERT mappings):

- SQL injection risks
- Buffer overflow concerns
- Dangerous function calls (`strcpy`, `gets`, etc.)
- File I/O vulnerabilities

Results are generated as an HTML report: `build/flawfinder-report.html`

### Interpreting Results

- **Level 5 (High):** Critical security issues — must be fixed
- **Level 3-4 (Medium):** Review and mitigate
- **Level 1-2 (Low):** Informational; usually acceptable

Current policy: **no Level 5 issues allowed in CI**.

## CI/CD Enforcement

### Regression Gate

The CI pipeline includes an automated regression gate:

1. **Cppcheck validation:** Compares report against approved suppressions
2. **Clang-Tidy checks:** Counts and reports violations (enforcement planned)
3. **Flawfinder enforcement:** Fails on Level 5 (high-risk) issues

### Workflow Steps

```yaml
- Run Cppcheck → Generate XML report
- Validate Cppcheck → Check against allowlist
  └─ FAIL if unapproved violations found
- Run Flawfinder → Generate HTML report
  └─ FAIL if Level 5 issues found
- Run Clang-Tidy → Capture and report violations
  └─ Report only (enforcement planned)
```

### Local Validation

Test your changes locally before pushing:

```bash
# Build with analysis targets
cd build && ninja analysis-cppcheck

# Validate suppressions are approved
python3 ../scripts/validate_static_analysis.py --report cppcheck-report.xml
```

## Handling CI Failures

### Cppcheck Violations Detected

If CI fails with "unapproved violations":

1. **Review the violations:**
   ```bash
   grep -A2 'id=' cppcheck-report.xml | head -20
   ```

2. **Investigate each finding:**
   - Is it a real bug? → Fix it
   - Is it a false-positive? → Add inline suppression + update allowlist
   - Is it third-party noise? → Add file-level suppression

3. **Document and commit:**
   - Update `scripts/validate_static_analysis.py`
   - Update `CHANGELOG.md` with suppression rationale
   - Commit with explanation

### Clang-Tidy Violations

Clang-Tidy violations currently report but don't block:

1. Review the violations in artifacts
2. Fix high-priority ones
3. For acceptable patterns: add clang-tidy comment suppression:
   ```cpp
   // NOLINT(clang-analyzer-core.NullDereference)
   int* ptr = nullptr;  // Intentional for testing
   ```

## Performance & Consistency

### Suppress Static Analysis Warnings On Vendor Code

All vendor/third-party targets have analysis disabled:

```cmake
# CMakeLists.txt shows pattern:
set_target_properties(vendor_lib PROPERTIES
    C_CLANG_TIDY ""
    CXX_CLANG_TIDY ""
)
```

This prevents noise from external dependencies.

### Reproducibility

Analysis results are deterministic when:
- Same cppcheck/clang-tidy versions
- Same compiler flags (`--std=c++20`)
- Same source code

Docker CI ensures consistency across environments.

## Future Enhancements

- [ ] **Clang-Tidy baseline tracking** — Fail on new violations only
- [ ] **SonarQube integration** — Long-term trend analysis
- [ ] **Automated fixing** — Apply clang-tidy fixes in CI
- [ ] **Third-party header whitelisting** — Reduce false-positives

## References

- [Cppcheck Documentation](http://cppcheck.sourceforge.net/)
- [Clang-Tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [Flawfinder Guide](https://dwheeler.com/flawfinder/)
- [CERT/CWE Coding Standards](https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+Coding+Standards)
