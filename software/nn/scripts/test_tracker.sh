#!/bin/bash

# Test Tracker - Runs tests individually and logs which ones pass/fail
# Used to identify which tests cause VSCode crashes

WORKSPACE="/home/ensismoebius/Repos/doutorado/software/nn"
BUILD_DIR="${WORKSPACE}/build"
TEST_BINARY="${BUILD_DIR}/src/core/layers/tests/layers_gtest"
RESULTS_FILE="${WORKSPACE}/TEST_RESULTS.log"
TRACKER_FILE="${WORKSPACE}/test_tracker.txt"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize tracker file with header
cat > "$TRACKER_FILE" << 'EOF'
# Test Tracker - Individual Test Results
# Format: TEST_NAME | STATUS (PASS/FAIL/CRASH) | TIMESTAMP

# LEGEND:
# PASS  = Test executed successfully
# FAIL  = Test executed but assertion failed
# CRASH = Test caused process to crash or hang
# SKIP  = Test skipped

# ============================================================================
# TEST RESULTS:
# ============================================================================

EOF

echo "Starting individual test tracking..."
echo "Results will be saved to: $TRACKER_FILE"
echo ""

# Get list of all tests
echo "Fetching test list..."
TESTS=$($TEST_BINARY --gtest_list_tests 2>/dev/null | grep -E "^\s+" | sed 's/^\s*//' | head -50)

if [ -z "$TESTS" ]; then
    echo "❌ Could not fetch test list. Trying alternative method..."
    # Fallback: use known tests
    TESTS=(
        "SimpleSignalOperationsTest.TestAMDF"
        "MSELossTest.ForwardAndBackward"
        "SequentialTest.ForwardAndBackward"
        "LinearLayerTest.ForwardSimple"
        "LeakyLayerTest.ForwardSpikeAndReset"
        "L1RegularizationTest.Forward"
        "Conv2dTest.ForwardAndBackward"
        "LayerExceptionTest.LinearInvalidDimensions"
        "LayerExceptionTest.Conv2dInvalidInputs"
        "LayerExceptionTest.SequentialEmptyLayers"
        "LayerExceptionTest.MSELossInvalidTargets"
    )
else
    # Convert to array
    mapfile -t TESTS_ARRAY <<<"$TESTS"
    TESTS=("${TESTS_ARRAY[@]}")
fi

echo "Found ${#TESTS[@]} tests to run"
echo ""

PASSED=0
FAILED=0
CRASHED=0
TOTAL=0

# Run each test individually
for TEST in "${TESTS[@]}"; do
    if [ -z "$TEST" ]; then
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    TEST_NAME=$(echo "$TEST" | xargs)
    
    echo -n "[$TOTAL] Testing: $TEST_NAME ... "
    
    # Run test with timeout
    TIMEOUT=10
    OUTPUT=$(timeout $TIMEOUT "$TEST_BINARY" --gtest_filter="$TEST_NAME" 2>&1)
    EXIT_CODE=$?
    
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}PASS${NC}"
        echo "$TEST_NAME | PASS | $TIMESTAMP" >> "$TRACKER_FILE"
        PASSED=$((PASSED + 1))
    elif [ $EXIT_CODE -eq 124 ]; then
        echo -e "${YELLOW}CRASH/TIMEOUT${NC}"
        echo "$TEST_NAME | CRASH | $TIMESTAMP" >> "$TRACKER_FILE"
        CRASHED=$((CRASHED + 1))
        echo "⚠️  Test timed out or crashed: $TEST_NAME"
    else
        echo -e "${RED}FAIL${NC}"
        echo "$TEST_NAME | FAIL | $TIMESTAMP" >> "$TRACKER_FILE"
        FAILED=$((FAILED + 1))
        # Show last few lines of output for debugging
        if [ -n "$OUTPUT" ]; then
            echo "     Last output: $(echo "$OUTPUT" | tail -1)"
        fi
    fi
    
    # Small delay between tests to avoid overwhelming the system
    sleep 0.5
done

# Print summary
echo ""
echo "============================================================================"
echo "TEST SUMMARY"
echo "============================================================================"
echo -e "Total Tests:  ${TOTAL}"
echo -e "✅ Passed:    ${GREEN}${PASSED}${NC}"
echo -e "❌ Failed:    ${RED}${FAILED}${NC}"
echo -e "⚠️  Crashed:   ${YELLOW}${CRASHED}${NC}"
echo ""
echo "Results saved to: $TRACKER_FILE"
echo ""

# Show crashed tests if any
CRASH_COUNT=$(grep -c " | CRASH | " "$TRACKER_FILE")
if [ "$CRASH_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  TESTS THAT CAUSED CRASHES:${NC}"
    grep " | CRASH | " "$TRACKER_FILE" | awk '{print "  - " $1}'
    echo ""
fi

# Show failed tests if any
FAIL_COUNT=$(grep -c " | FAIL | " "$TRACKER_FILE")
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo -e "${RED}❌ TESTS THAT FAILED:${NC}"
    grep " | FAIL | " "$TRACKER_FILE" | awk '{print "  - " $1}'
    echo ""
fi

echo "View detailed results with: cat $TRACKER_FILE"
