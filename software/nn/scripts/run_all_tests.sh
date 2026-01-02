#!/bin/bash

# Comprehensive Test Runner with Crash Detection
# Runs all tests and captures output, timeouts, and crashes
# Useful for identifying problematic tests when VSCode crashes

WORKSPACE="/home/ensismoebius/Repos/doutorado/software/nn"
BUILD_DIR="${WORKSPACE}/build"
RESULTS_DIR="${WORKSPACE}/test_results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Create results directory
mkdir -p "$RESULTS_DIR"

# Log files
FULL_LOG="$RESULTS_DIR/test_run_${TIMESTAMP}.log"
CRASH_LOG="$RESULTS_DIR/crashes_${TIMESTAMP}.log"
SUMMARY_LOG="$RESULTS_DIR/summary_${TIMESTAMP}.txt"

echo "=========================================="
echo "COMPREHENSIVE TEST RUNNER"
echo "=========================================="
echo "Workspace: $WORKSPACE"
echo "Build Dir: $BUILD_DIR"
echo "Results saved to: $RESULTS_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Function to run a single test binary
run_test_binary() {
    local binary=$1
    local test_name=$2
    local timeout=$3
    
    echo "Running: $test_name (binary: $(basename $binary))" | tee -a "$FULL_LOG"
    
    # Run with timeout and capture both stdout and stderr
    timeout $timeout "$binary" 2>&1 | tee -a "$FULL_LOG"
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "⚠️  CRASH/TIMEOUT: $test_name" | tee -a "$CRASH_LOG"
        return 1
    elif [ $exit_code -eq 0 ]; then
        echo "✅ PASS: $test_name"
        return 0
    else
        echo "❌ FAIL: $test_name (exit code: $exit_code)" | tee -a "$FULL_LOG"
        return 1
    fi
}

# Initialize log files
cat > "$FULL_LOG" << EOF
TEST RUN LOG - $TIMESTAMP
=====================================

This file contains the full output of all test runs.
EOF

cat > "$CRASH_LOG" << EOF
CRASH/TIMEOUT LOG - $TIMESTAMP
=====================================

Tests that timed out or crashed:
EOF

# Find all test binaries
echo "Finding test binaries..."
TEST_BINARIES=($(find "$BUILD_DIR" -name "*_gtest" -o -name "*_test" -type f 2>/dev/null | sort))

if [ ${#TEST_BINARIES[@]} -eq 0 ]; then
    echo "❌ No test binaries found in $BUILD_DIR"
    exit 1
fi

echo "Found ${#TEST_BINARIES[@]} test binaries"
echo ""

# Track results
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_CRASH=0

# Run each test binary
for binary in "${TEST_BINARIES[@]}"; do
    test_name=$(basename "$binary")
    
    echo "================================"
    echo "Testing: $test_name"
    echo "================================"
    
    if run_test_binary "$binary" "$test_name" 60; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        TOTAL_CRASH=$((TOTAL_CRASH + 1))
    fi
    
    echo ""
done

# Generate summary
cat > "$SUMMARY_LOG" << EOF
TEST RUN SUMMARY - $TIMESTAMP
=====================================

Test Binaries Run: ${#TEST_BINARIES[@]}
Passed:            $TOTAL_PASS
Crashed/Failed:    $TOTAL_CRASH

Success Rate:      $(( TOTAL_PASS * 100 / ${#TEST_BINARIES[@]} ))%

Binaries tested:
EOF

for binary in "${TEST_BINARIES[@]}"; do
    echo "  - $(basename $binary)" >> "$SUMMARY_LOG"
done

# Print summary to console
echo ""
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
cat "$SUMMARY_LOG"
echo ""

# Check for crashes
CRASH_COUNT=$(grep -c "CRASH/TIMEOUT" "$CRASH_LOG")
if [ "$CRASH_COUNT" -gt 0 ]; then
    echo "⚠️  CRASHES DETECTED!"
    echo ""
    grep "CRASH/TIMEOUT" "$CRASH_LOG"
fi

echo ""
echo "Full results available in:"
echo "  - $FULL_LOG"
echo "  - $CRASH_LOG"
echo "  - $SUMMARY_LOG"
echo ""
