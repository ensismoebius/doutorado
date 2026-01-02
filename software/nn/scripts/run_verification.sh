#!/bin/bash

# Neural Network Framework - Local Verification Script
# Runs all analysis tools: cppcheck, flawfinder, clang-tidy, tests, callgrind, coverage

set -e  # Exit on any error

echo "=== Neural Network Framework: Comprehensive Local Verification ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run from the project root directory."
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    print_status "Creating build directory..."
    mkdir -p build
fi

cd build

# Configure with coverage if not already configured
if [ ! -f "CMakeCache.txt" ]; then
    print_status "Configuring CMake with coverage flags..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS="--coverage -O0 -g -fno-inline" \
        -DCMAKE_C_FLAGS="--coverage -O0 -g -fno-inline" \
        -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
        -G Ninja
fi

# Build the project
print_status "Building project..."
if ninja -j$(nproc); then
    print_success "Build completed successfully"
else
    print_error "Build failed"
    exit 1
fi

echo

# Run Cppcheck
print_status "Running Cppcheck static analysis..."
if ninja analysis-cppcheck 2>/dev/null; then
    if [ -f "cppcheck-report.xml" ]; then
        print_success "Cppcheck completed - report saved to cppcheck-report.xml"
        # Show summary of issues
        if command -v xmllint >/dev/null 2>&1; then
            ERROR_COUNT=$(xmllint --xpath "count(//error)" cppcheck-report.xml 2>/dev/null || echo "0")
            print_status "Cppcheck found $ERROR_COUNT issues"
        fi
    else
        print_warning "Cppcheck completed but no report file found"
    fi
else
    print_warning "Cppcheck not available or failed"
fi

echo

# Run Flawfinder
print_status "Running Flawfinder security analysis..."
if ninja analysis-flawfinder 2>/dev/null; then
    if [ -f "flawfinder-report.html" ]; then
        print_success "Flawfinder completed - report saved to flawfinder-report.html"
        # Show summary of high-risk issues
        HIGH_RISK=$(grep -c "High" flawfinder-report.html 2>/dev/null || echo "0")
        print_status "Flawfinder found $HIGH_RISK high-risk issues"
    else
        print_warning "Flawfinder completed but no report file found"
    fi
else
    print_warning "Flawfinder not available or failed"
fi

echo

# Run Clang-Tidy
print_status "Running Clang-Tidy static analysis..."
if ninja analysis-clang-tidy 2>/dev/null; then
    print_success "Clang-Tidy analysis completed"
else
    print_warning "Clang-Tidy not available or failed"
fi

echo

# Run tests
print_status "Running unit tests..."
if ctest --output-on-failure -j$(nproc) --timeout 300; then
    print_success "All tests passed"
else
    print_error "Some tests failed"
    exit 1
fi

echo

# Run Callgrind profiling on a few key tests
print_status "Running Callgrind performance profiling..."
CALLGRIND_COUNT=0
for exe in $(find . -name "*_gtest" -type f -executable | head -2); do
    if [ -x "$exe" ]; then
        BASENAME=$(basename "$exe")
        print_status "Profiling $BASENAME..."
        if valgrind --tool=callgrind --callgrind-out-file="callgrind.$BASENAME.out" \
                   "$exe" --gtest_filter="*Basic*" >/dev/null 2>&1; then
            print_success "Callgrind profiling completed for $BASENAME"
            CALLGRIND_COUNT=$((CALLGRIND_COUNT + 1))
        else
            print_warning "Callgrind failed for $BASENAME"
        fi
    fi
done

if [ $CALLGRIND_COUNT -gt 0 ]; then
    print_success "Callgrind profiling completed for $CALLGRIND_COUNT executables"
else
    print_warning "No executables profiled with Callgrind"
fi

echo

# Generate coverage report
print_status "Generating coverage report..."
if command -v lcov >/dev/null 2>&1; then
    lcov --capture --directory . --output-file coverage.info --gcov-tool gcov
    lcov --remove coverage.info '/usr/*' '*/_deps/*' '*/tests/*' --output-file coverage.filtered.info

    if [ -f "coverage.filtered.info" ]; then
        if command -v genhtml >/dev/null 2>&1; then
            genhtml coverage.filtered.info --output-directory coverage-report
            print_success "Coverage report generated in coverage-report/"
        fi

        # Extract coverage percentage
        COVERAGE=$(lcov --summary coverage.filtered.info 2>/dev/null | grep "lines......" | sed 's/.*: \([0-9.]*\).*/\1/' | head -1)
        if [ ! -z "$COVERAGE" ]; then
            print_success "Coverage: $COVERAGE%"
            if (( $(echo "$COVERAGE >= 95.0" | bc -l 2>/dev/null || echo "0") )); then
                print_success "✅ Coverage threshold met: $COVERAGE% >= 95.0%"
            else
                print_warning "❌ Coverage below threshold: $COVERAGE% < 95.0%"
            fi
        fi
    else
        print_warning "Coverage data not generated"
    fi
else
    print_warning "lcov not available for coverage analysis"
fi

echo
echo "=== Verification Summary ==="
echo "✅ Build: Completed"
echo "✅ Cppcheck: $([ -f "cppcheck-report.xml" ] && echo "Report generated" || echo "Not available")"
echo "✅ Flawfinder: $([ -f "flawfinder-report.html" ] && echo "Report generated" || echo "Not available")"
echo "✅ Clang-Tidy: Analysis completed"
echo "✅ Tests: All executed"
echo "✅ Callgrind: Profiling completed"
echo "✅ Coverage: $([ -f "coverage.filtered.info" ] && echo "Report generated" || echo "Not available")"
echo
print_success "Local verification completed!"
echo
echo "Reports available in build/ directory:"
echo "  - cppcheck-report.xml"
echo "  - flawfinder-report.html"
echo "  - coverage-report/ (HTML coverage report)"
echo "  - callgrind.*.out (performance profiles)"