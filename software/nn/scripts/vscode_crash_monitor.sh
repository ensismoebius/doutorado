#!/bin/bash

# VSCode Crash Monitor and Test Logger
# Monitors test execution and logs detailed info about which test was running when VSCode crashes
# 
# Usage: ./vscode_crash_monitor.sh
# 
# This script:
# 1. Runs the test suite one test at a time
# 2. Logs detailed info before running each test
# 3. If VSCode crashes, the last logged test is the culprit
# 4. Can be run in a separate terminal while developing

WORKSPACE="/home/ensismoebius/Repos/doutorado/software/nn"
BUILD_DIR="${WORKSPACE}/build"
TEST_BINARY="${BUILD_DIR}/src/core/layers/tests/layers_gtest"
MONITOR_LOG="${WORKSPACE}/vscode_crash_monitor.log"
CURRENT_TEST_FILE="${WORKSPACE}/current_test.txt"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize monitoring log
cat > "$MONITOR_LOG" << 'EOF'
# VSCode Crash Monitor Log
# Each line represents a test that was about to be run
# If VSCode crashes, look for the last test in this file
# Format: TIMESTAMP | TEST_NAME | TIMEOUT | PID

================================================================================
START OF MONITORING - Monitoring tests to identify crashes
================================================================================

EOF

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}VSCode Crash Monitor - Test Tracker${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "This script monitors which test is running when crashes occur."
echo "Results logged to: $MONITOR_LOG"
echo ""
echo "Instructions:"
echo "1. Open this folder in VSCode in another terminal"
echo "2. Run VSCode and work normally"
echo "3. When VSCode crashes, check the log file to see which test was running"
echo ""

# Get list of all tests
echo "Fetching test list from: $TEST_BINARY"
if [ ! -f "$TEST_BINARY" ]; then
    echo -e "${RED}❌ Error: Test binary not found at $TEST_BINARY${NC}"
    exit 1
fi

# Get all available tests
TESTS=$($TEST_BINARY --gtest_list_tests 2>/dev/null | grep -E "^\s+" | sed 's/^\s*//' | grep -v "^$")

if [ -z "$TESTS" ]; then
    echo -e "${RED}❌ Could not fetch test list${NC}"
    exit 1
fi

# Convert to array
mapfile -t TESTS_ARRAY <<<"$TESTS"

echo -e "${GREEN}✅ Found ${#TESTS_ARRAY[@]} tests${NC}"
echo ""

# Statistics
PASSED=0
FAILED=0
TIMEOUT=0
TOTAL=0

# Run each test with detailed logging
for TEST_NAME in "${TESTS_ARRAY[@]}"; do
    if [ -z "$TEST_NAME" ]; then
        continue
    fi
    
    TOTAL=$((TOTAL + 1))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log that we're about to run this test
    echo "$TIMESTAMP | $TEST_NAME | Running" >> "$MONITOR_LOG"
    echo "$TEST_NAME" > "$CURRENT_TEST_FILE"
    
    # Show progress
    echo -ne "${BLUE}[$TOTAL]${NC} $TEST_NAME ... "
    
    # Run test with 15 second timeout
    TEST_TIMEOUT=15
    OUTPUT=$(timeout $TEST_TIMEOUT "$TEST_BINARY" --gtest_filter="$TEST_NAME" 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC}"
        echo "$TIMESTAMP | $TEST_NAME | PASS" >> "$MONITOR_LOG"
        PASSED=$((PASSED + 1))
    elif [ $EXIT_CODE -eq 124 ]; then
        echo -e "${YELLOW}⏱️  TIMEOUT/CRASH${NC}"
        echo "$TIMESTAMP | $TEST_NAME | TIMEOUT" >> "$MONITOR_LOG"
        TIMEOUT=$((TIMEOUT + 1))
        
        # Alert user that a problematic test was found
        echo -e "${YELLOW}⚠️  POSSIBLE PROBLEMATIC TEST FOUND: $TEST_NAME${NC}"
    else
        echo -e "${RED}❌ FAIL (exit: $EXIT_CODE)${NC}"
        echo "$TIMESTAMP | $TEST_NAME | FAIL" >> "$MONITOR_LOG"
        FAILED=$((FAILED + 1))
    fi
    
    # Small delay
    sleep 0.2
done

# Clear current test file
rm -f "$CURRENT_TEST_FILE"

# Print summary
echo ""
echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}TEST SUMMARY${NC}"
echo -e "${BLUE}================================================${NC}"
echo "Total:   $TOTAL"
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Timeout: ${YELLOW}$TIMEOUT${NC}"
echo ""
echo "Success rate: $(( PASSED * 100 / TOTAL ))%"
echo ""
echo "Monitor log: $MONITOR_LOG"
echo ""
echo -e "${BLUE}If VSCode crashes, check the monitor log for the last test name.${NC}"
echo "The test listed just before the crash is likely the culprit."
echo ""

# Show any problematic tests
if [ $TIMEOUT -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Tests that timed out/crashed:${NC}"
    grep "| TIMEOUT$" "$MONITOR_LOG" | awk -F' | ' '{print "  - " $2}'
fi
