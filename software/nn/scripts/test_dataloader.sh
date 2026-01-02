#!/bin/bash
cd /home/ensismoebius/Repos/doutorado/software/nn/build

echo "=== Building dataLoaders_gtest ==="
ninja dataLoaders_gtest
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    echo "Build failed with exit code $BUILD_EXIT"
    exit $BUILD_EXIT
fi

echo ""
echo "=== Running exception tests ==="
./src/core/dataLoaders/tests/dataLoaders_gtest --gtest_filter="DataLoaderExceptionTest.*"
echo ""

echo "=== Running iterator independence test ==="
./src/core/dataLoaders/tests/dataLoaders_gtest --gtest_filter="DataLoaderThreadSafetyTest.IteratorIndependence"
echo ""

echo "=== Running all dataLoader tests ==="
ctest --test-dir . -R dataLoaders_gtest --output-on-failure
