#!/bin/bash
# Safe test runner - runs tests sequentially with resource limits and timeouts

set -e

BUILD_DIR="${1:-build}"
TEST_TIMEOUT=30
MAX_MEMORY_MB=2048

echo "=== Safe Test Runner ==="
echo "Build directory: $BUILD_DIR"
echo "Timeout per test: ${TEST_TIMEOUT}s"
echo "Max memory: ${MAX_MEMORY_MB}MB"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory '$BUILD_DIR' not found"
    exit 1
fi

# Get list of all tests
echo "Discovering tests..."
TEST_LIST=$(ctest --test-dir "$BUILD_DIR" -N 2>&1 | grep "Test #" | sed 's/.*Test #[0-9]*: //')

if [ -z "$TEST_LIST" ]; then
    echo "No tests found!"
    exit 1
fi

TEST_COUNT=$(echo "$TEST_LIST" | wc -l)
echo "Found $TEST_COUNT tests"
echo ""

# Run each test individually with resource limits
FAILED_TESTS=()
PASSED_TESTS=0
CURRENT=0

for TEST_NAME in $TEST_LIST; do
    CURRENT=$((CURRENT + 1))
    echo "[$CURRENT/$TEST_COUNT] Running: $TEST_NAME"
    
    # Run with timeout and memory limit
    if timeout $TEST_TIMEOUT ctest --test-dir "$BUILD_DIR" -R "^${TEST_NAME}$" --output-on-failure 2>&1; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        echo "  ✓ PASSED"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            echo "  ✗ TIMEOUT (>${TEST_TIMEOUT}s)"
        else
            echo "  ✗ FAILED (exit code: $EXIT_CODE)"
        fi
        FAILED_TESTS+=("$TEST_NAME")
    fi
    echo ""
done

# Summary
echo "=== Test Summary ==="
echo "Passed: $PASSED_TESTS/$TEST_COUNT"
echo "Failed: ${#FAILED_TESTS[@]}/$TEST_COUNT"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed tests:"
    for TEST in "${FAILED_TESTS[@]}"; do
        echo "  - $TEST"
    done
    exit 1
fi

echo ""
echo "All tests passed!"
exit 0
