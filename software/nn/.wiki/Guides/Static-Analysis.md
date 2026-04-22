# Static Analysis & Code Quality

## Overview

The nn project uses a comprehensive static analysis pipeline:

- **Cppcheck** — Detects bugs, dead code, coding style issues
- **Clang-Tidy** — Enforces modern C++ practices, catches performance issues
- **Flawfinder** — Identifies security-sensitive code patterns

## Cppcheck

### Running

```bash
cd build
ninja analysis-cppcheck
```

### Known Suppressions

Approved suppressions for third-party header noise and valid runtime guards:

| File | Issue | Reason |
|------|-------|--------|
| `BatchPrefetcher.cpp` | `knownConditionTrueFalse` | Valid runtime guard checking `stop_requested_` |
| `wave_gtest.cpp` | `passedByValue` | Callback parameter contract fixed by definition |
| `speaker_demo.cpp` | `useStlAlgorithm` | Argparse header template instantiation |
| `Config.cpp` | Various | Nlohmann_json template noise |

List approved suppressions:
```bash
python3 scripts/validate_static_analysis.py --list-approved
```

### Adding Suppressions

1. Verify the issue is a false-positive
2. Add inline suppression:
   ```cpp
   // cppcheck-suppress knownConditionTrueFalse
   if (stop_requested_) { break; }
   ```
3. Update `scripts/validate_static_analysis.py`

## Clang-Tidy

### Running

During compilation:
```bash
cmake -S . -B build -DNN_ENABLE_CLANG_TIDY=ON
ninja -j$(nproc)
```

Standalone:
```bash
cd build
ninja analysis-clang-tidy
```

### Configuration

Located in `.clang-tidy`:
- Core safety: `bugprone-*`, `clang-analyzer-*`
- Modern C++: `modernize-*`
- Performance: `performance-*`

## Flawfinder

Generates HTML report: `build/flawfinder-report.html`

Interpretation:
- Level 5: Critical — must fix
- Level 3-4: Medium — review and mitigate
- Level 1-2: Low — informational

Policy: **No Level 5 issues allowed in CI**

## CI Enforcement

```yaml
- Run Cppcheck → Generate XML
- Validate → Check against allowlist (FAIL if unapproved)
- Run Flawfinder → FAIL on Level 5
- Run Clang-Tidy → Report only (enforcement planned)
```

## Local Validation

```bash
cd build && ninja analysis-cppcheck
python3 ../scripts/validate_static_analysis.py --report cppcheck-report.xml
```

## See Also

- [Architecture](../Architecture.md)
- [Device](../Core/Device.md) (for OpenCL backend analysis)