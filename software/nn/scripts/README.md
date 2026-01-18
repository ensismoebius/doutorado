#!/bin/bash

# scripts/README.md - Test Script Guide

## Available Scripts

### vscode_crash_monitor.sh

**Purpose**: Real-time test monitoring to identify crashes

**How it works**:

- Runs tests one by one
- Logs timestamp of each test
- Updates `current_test.txt` with currently running test
- If VSCode crashes, you can see which test was running

**Usage**:

```bash
# Run in a separate terminal while developing
./scripts/vscode_crash_monitor.sh
```

**Output**:

- `vscode_crash_monitor.log` - Detailed test log with timestamps

**When to use**: When VSCode keeps crashing and you need to find the problematic test

---

### test_tracker.sh

**Purpose**: Quick individual test tracking

**How it works**:

- Runs tests individually
- Marks each as PASS, FAIL, or CRASH
- Saves results to `test_tracker.txt`
- Shows summary with pass rate

**Usage**:

```bash
./scripts/test_tracker.sh
```

**Output**:

- `test_tracker.txt` - Per-test results with timestamps

**When to use**: When you want a quick overview of which tests pass/fail

---

### run_all_tests.sh

**Purpose**: Comprehensive test suite runner with logging

**How it works**:

- Finds all test binaries in build directory
- Runs each with 60-second timeout
- Captures all output to log files
- Generates summary report

**Usage**:

```bash
./scripts/run_all_tests.sh
```

**Output**:

- `test_results/test_run_YYYYMMDD_HHMMSS.log` - Full output
- `test_results/crashes_YYYYMMDD_HHMMSS.log` - Crashes/timeouts
- `test_results/summary_YYYYMMDD_HHMMSS.txt` - Summary

**When to use**: When you need comprehensive logging and documentation of test results

---

## Quick Start

1. **Monitor tests while developing**:

   ```bash
   # Terminal 1
   ./scripts/vscode_crash_monitor.sh

   # Terminal 2
   code .
   ```

   If VSCode crashes, check Terminal 1 for the problematic test.

2. **Quick test check**:

   ```bash
   ./scripts/test_tracker.sh
   ```

3. **Full test documentation**:
   ```bash
   ./scripts/run_all_tests.sh
   ```

## Making Scripts Executable

If scripts don't run, make them executable:

```bash
chmod +x ./vscode_crash_monitor.sh
chmod +x ./test_tracker.sh
chmod +x ./run_all_tests.sh
```

## Current Test Status

✅ All 40+ tests passing

See `TEST_SUITE_COMPLETE.md` for full details.
