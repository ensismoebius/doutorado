# Quick Reference - Test & Debug Commands

## Running Tests

### All Tests (Recommended)

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn
./scripts/vscode_crash_monitor.sh
```

Shows which test is running in real-time. If VSCode crashes, you'll know exactly which test caused it.

### Quick Test Check

```bash
cd build
ctest --output-on-failure -j4
```

Run all tests in parallel with failure details.

### Single Test

```bash
cd build
./src/core/layers/tests/layers_gtest --gtest_filter="TestName"
```

### Specific Test Suite

```bash
cd build
./src/core/layers/tests/layers_gtest --gtest_filter="LinearLayerTest*"
```

## Building

### Full Build

```bash
cd /home/ensismoebius/Repos/doutorado/software/nn
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

### Clean Build

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
```

## Debugging

### With AddressSanitizer Output

```bash
ASAN_OPTIONS=verbosity=2:halt_on_error=1 \
./build/src/core/layers/tests/layers_gtest --gtest_filter="TestName"
```

### With GDB

```bash
gdb ./build/src/core/layers/tests/layers_gtest
(gdb) run --gtest_filter="TestName"
(gdb) bt  # Print backtrace on crash
```

## Test Status

✅ **All 40+ tests passing**

Critical tests that were fixed:

- ✅ MSELossTest.ForwardAndBackward
- ✅ LinearLayerTest tests
- ✅ L1/L2Regularization tests
- ✅ SimpleResNetTest tests
- ✅ All 19 Conv2d tests

## Key Files

| File                              | Purpose                      |
| --------------------------------- | ---------------------------- |
| `TEST_SUITE_COMPLETE.md`          | Full test status report      |
| `TEST_TRACKING_GUIDE.md`          | How to use monitoring tools  |
| `vscode_crash_monitor.log`        | Real-time test execution log |
| `scripts/vscode_crash_monitor.sh` | Monitor script               |
| `scripts/test_tracker.sh`         | Quick test tracker           |
| `scripts/run_all_tests.sh`        | Full test suite runner       |

## If VSCode Crashes

1. Check which test was running:

   ```bash
   tail vscode_crash_monitor.log
   ```

2. Run that test individually to confirm:

   ```bash
   cd build
   ./src/core/layers/tests/layers_gtest --gtest_filter="ProblematicTest"
   ```

3. Check for memory issues:
   ```bash
   ASAN_OPTIONS=verbosity=2 \
   ./build/src/core/layers/tests/layers_gtest --gtest_filter="ProblematicTest"
   ```

## Common Issues

| Issue                                   | Solution                                                      |
| --------------------------------------- | ------------------------------------------------------------- |
| Tests timeout                           | Increase timeout or check for infinite loops                  |
| Memory errors                           | Run with ASAN_OPTIONS verbosity                               |
| Tests pass separately but fail in suite | May be test ordering issue                                    |
| VSCode crashes                          | Use vscode_crash_monitor.sh to identify culprit               |
| Build fails                             | Run `rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Debug` |

## Project Structure

```
/home/ensismoebius/Repos/doutorado/software/nn/
├── scripts/              # Test and utility scripts
├── build/                # CMake build directory
├── src/                  # Source code
│   └── core/
│       ├── layers/       # Neural network layers
│       ├── tensor/       # Tensor implementation
│       └── optimizers/   # Optimization algorithms
└── TEST_SUITE_COMPLETE.md  # This status report
```

## Current Status ✅

- **Build**: Compiling successfully
- **Tests**: All 40+ tests passing
- **Memory**: No leaks or corruption detected
- **Sanitizers**: AddressSanitizer/UBSan enabled and clean
- **VSCode**: Safe to use with test monitoring

---

**Last Updated**: 2026-01-02  
**Test Suite Status**: 🟢 ALL PASSING  
**Memory Status**: 🟢 CLEAN (No corruption/leaks)
