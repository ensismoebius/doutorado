# nn

## STL Commands Used In This Project

This section lists the commands used for static tooling, linting, build validation, testing, and coverage in this repository.

### CMake + Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DNN_ENABLE_CLANG_TIDY=ON
cmake --build build -j$(nproc)
```

### Ninja Analysis Targets

```bash
cd build
ninja analysis-cppcheck
ninja analysis-clang-tidy
ninja analysis-flawfinder
ninja analysis-all
```

### Cppcheck

```bash
cppcheck src/ --enable=warning,style,performance --suppress=missingIncludeSystem --std=c++20 --xml --output-file=cppcheck-test-report.xml

cppcheck src/ \
  --enable=all \
  --check-level=exhaustive \
  --std=c++20 \
  --platform=native \
  --suppress=missingIncludeSystem \
  --suppress=unusedFunction \
  --quiet \
  --xml \
  --output-file=cppcheck.xml
```

### Clang-Tidy

```bash
clang-tidy --config-file=.clang-tidy -p build src/core/myfile.cpp
```

### Flawfinder

```bash
flawfinder --minlevel=1 --columns --singleline src/
```

### Static Analysis Validation Script

```bash
python3 scripts/validate_static_analysis.py --list-approved
python3 scripts/validate_static_analysis.py --report build/cppcheck-report.xml
python3 scripts/validate_static_analysis.py --report cppcheck-test-report.xml
```

### Tests

```bash
cd build
ctest --output-on-failure -j$(nproc)
ctest --output-on-failure -j$(nproc) --timeout 300
```

### Coverage

```bash
cd build
lcov --capture --directory . --output-file coverage.info --gcov-tool gcov
lcov --remove coverage.info '/usr/*' '*/_deps/*' '*/tests/*' '*/BatchPrefetcher*' --output-file coverage.filtered.info
genhtml coverage.filtered.info --output-directory coverage-report
bash ../scripts/check_core_coverage_gate.sh coverage.filtered.info
```

### Profiling

```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.<exe>.out <exe> --gtest_filter='*Basic*'
```

### Useful Inspection Commands

```bash
grep 'file0="src/' cppcheck_lowrisk_current_inline.xml | grep -E 'id="(knownConditionTrueFalse|useStlAlgorithm|useInitializationList|passedByValue)"' | wc -l
head -50 cppcheck-test-report.xml | grep -E 'error|location' | head -20
```

## Notes

- CI uses the same analysis flow defined in .github/workflows/ci.yml.
- The approved static-analysis allowlist is maintained in scripts/validate_static_analysis.py.
