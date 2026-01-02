# Test Tracking and Crash Investigation Guide

## Overview

When VSCode crashes, it's important to identify which test is causing the problem. This guide explains how to use the test tracking tools to pinpoint the issue.

## Available Tools

### 1. **test_tracker.sh** - Quick Individual Test Runner

Runs tests one at a time and logs results to `test_tracker.txt`

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn
./scripts/test_tracker.sh
```

**What it does:**

- Runs each test individually
- Marks each test as PASS, FAIL, or CRASH
- Shows a summary with pass rate
- Saves detailed results to `test_tracker.txt`

**Use this when:**

- You want to quickly identify which tests crash
- You want a simple per-test pass/fail report
- Testing specific test groups

**Output files:**

- `test_tracker.txt` - Detailed per-test results with timestamps

---

### 2. **run_all_tests.sh** - Full Test Suite Runner

Runs all test binaries with comprehensive logging

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn
./scripts/run_all_tests.sh
```

**What it does:**

- Finds all test binaries in the build directory
- Runs each binary with 60-second timeout
- Captures all output to log files
- Generates summary report

**Use this when:**

- Running a full project test suite
- Need detailed logs for all tests
- Investigating complete test status

**Output files:**

- `test_results/test_run_YYYYMMDD_HHMMSS.log` - Full test output
- `test_results/crashes_YYYYMMDD_HHMMSS.log` - Crash/timeout list
- `test_results/summary_YYYYMMDD_HHMMSS.txt` - Summary report

---

### 3. **vscode_crash_monitor.sh** - Continuous Monitor (Recommended for VSCode Crashes)

Runs tests continuously and logs which test is running in real-time

```bash
# Run in a separate terminal while VSCode is open
cd /home/ensismoebius/Repos/doutorado/software/nn
./scripts/vscode_crash_monitor.sh
```

**What it does:**

- Runs tests one by one with detailed logging
- Records timestamp for each test
- Creates `current_test.txt` showing which test is currently running
- If VSCode crashes, you can check which test was the culprit

**Use this when:**

- VSCode keeps crashing
- You want real-time tracking of test execution
- Need to identify the exact problematic test

**How to use it to find VSCode crash cause:**

1. Open two terminals
2. In terminal 1: Start the monitor
   ```bash
   ./scripts/vscode_crash_monitor.sh
   ```
3. In terminal 2: Open and use VSCode
   ```bash
   code .
   ```
4. Use VSCode normally until it crashes
5. Go back to terminal 1 and check the last logged test
6. That test is likely causing the crash

**Output files:**

- `vscode_crash_monitor.log` - Real-time test execution log
- `current_test.txt` - Currently running test (updated in real-time)

---

## Quick Crash Investigation Steps

### When VSCode crashes:

1. **Check current_test.txt** (if using vscode_crash_monitor.sh)

   ```bash
   cat current_test.txt
   # Shows which test was running
   ```

2. **Check the monitor log** for last few tests

   ```bash
   tail -20 vscode_crash_monitor.log
   # Shows last 20 test results
   ```

3. **Run the problematic test individually** to verify

   ```bash
   cd build
   ./src/core/layers/tests/layers_gtest --gtest_filter="TestName"
   ```

4. **Check for AddressSanitizer output**
   The test binary is compiled with ASan/UBSan, so any memory issues will be reported

5. **Run with verbose output**
   ```bash
   ASAN_OPTIONS=verbosity=2 ./build/src/core/layers/tests/layers_gtest --gtest_filter="TestName"
   ```

---

## Expected Results

### Current Test Status (as of latest run):

Looking at the test tracker output, the following tests are known to **PASS** ✅:

- SimpleSignalOperationsTest.TestAMDF
- MSELossTest.ForwardAndBackward
- SequentialTest.ForwardAndBackward
- LinearLayerTest.ForwardSimple
- LeakyLayerTest tests
- L1/L2 Regularization tests
- Conv2dTest (all variants)
- LayerExceptionTest (all validation tests)
- SimpleResNetTest tests

If VSCode crashes when a different test runs, that test is the issue.

---

## Debugging Tips

### If a test causes VSCode to crash:

1. **Run it outside VSCode first** to see if it's the test or VSCode

   ```bash
   timeout 10 ./build/src/core/layers/tests/layers_gtest --gtest_filter="ProblematicTest"
   ```

2. **Check for memory issues** with verbose AddressSanitizer

   ```bash
   ASAN_OPTIONS=verbosity=2:halt_on_error=1 \
   ./build/src/core/layers/tests/layers_gtest --gtest_filter="ProblematicTest"
   ```

3. **Run with debugging symbols** (already enabled in Debug build)

   ```bash
   gdb ./build/src/core/layers/tests/layers_gtest
   (gdb) run --gtest_filter="ProblematicTest"
   ```

4. **Check CMake build configuration**
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Debug
   cmake --build build -j$(nproc)
   ```

---

## File Locations

| File                       | Purpose                 | Updated By                |
| -------------------------- | ----------------------- | ------------------------- |
| `test_tracker.txt`         | Individual test results | `test_tracker.sh`         |
| `vscode_crash_monitor.log` | Real-time monitor log   | `vscode_crash_monitor.sh` |
| `current_test.txt`         | Currently running test  | `vscode_crash_monitor.sh` |
| `test_results/`            | Full test suite logs    | `run_all_tests.sh`        |

---

## Common Issues and Solutions

### Issue: Tests timeout but don't crash

**Solution**: May be stuck in infinite loop. Increase timeout or add debug output to the test.

### Issue: Different tests fail on different runs

**Solution**: Could be a race condition. Run tests serially (not parallel) or check for thread safety.

### Issue: VSCode crashes but test passes when run separately

**Solution**: VSCode may have limited memory. Check system resources with `top` or `free`.

### Issue: Can't find problematic test

**Solution**: Run `vscode_crash_monitor.sh` in a separate terminal while using VSCode. When it crashes, the monitor will show exactly which test was running.

---

## Next Actions

After identifying the problematic test:

1. Create a minimal reproduction case
2. Review the test code for memory issues
3. Check for unininitialized variables
4. Look for buffer overflows or out-of-bounds access
5. Review any recent changes to the test or code under test

---

## Example Workflow

```bash
# 1. Start monitoring in one terminal
cd /home/ensismoebius/Repos/doutorado/software/nn
./scripts/vscode_crash_monitor.sh

# 2. In another terminal, open VSCode
code .

# 3. When VSCode crashes, check the monitor
tail -5 vscode_crash_monitor.log

# 4. Example output showing last test:
# 2026-01-02 04:23:50 | Conv2dTest.BiasShapeVariants | PASS
# 2026-01-02 04:23:51 | LayerExceptionTest.LinearInvalidDimensions | Running
# (VSCode crashes here)

# 5. Run that test individually
cd build
./src/core/layers/tests/layers_gtest --gtest_filter="LayerExceptionTest.LinearInvalidDimensions"

# 6. If it passes, VSCode has limited resources
# If it fails/crashes, found the issue!
```
