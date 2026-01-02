#!/bin/bash
# Test diagnostics - identify memory hogs and problematic tests

set -e

BUILD_DIR="${1:-build}"

echo "=== Test Diagnostics ==="
echo "Build directory: $BUILD_DIR"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Check available memory
echo "System Memory:"
free -h
echo ""

# Check for test executables
echo "Test executables:"
find "$BUILD_DIR" -name "*gtest" -type f -executable 2>/dev/null | while read -r TEST_EXE; do
    SIZE=$(du -h "$TEST_EXE" | cut -f1)
    echo "  $SIZE - $TEST_EXE"
done
echo ""

# Try to list tests without running them
echo "Attempting to discover tests..."
if timeout 5 ctest --test-dir "$BUILD_DIR" -N 2>&1 | tee /tmp/ctest_list.txt; then
    TEST_COUNT=$(grep -c "Test #" /tmp/ctest_list.txt || echo "0")
    echo ""
    echo "Found $TEST_COUNT tests"
    
    if [ "$TEST_COUNT" -gt 0 ]; then
        echo ""
        echo "Test list:"
        grep "Test #" /tmp/ctest_list.txt | head -20
    fi
else
    echo "ERROR: ctest failed to list tests (timeout or crash)"
    echo "This suggests a configuration issue or immediate crash on startup"
    exit 1
fi

echo ""
echo "=== Recommendations ==="
echo "1. Run tests individually: ./scripts/safe_test_runner.sh"
echo "2. Check for memory leaks: valgrind <test_executable>"
echo "3. Monitor resources: watch -n1 'ps aux | grep test'"
echo "4. Reduce parallelism: ctest --test-dir build -j 1"
echo "5. Enable core dumps: ulimit -c unlimited"
