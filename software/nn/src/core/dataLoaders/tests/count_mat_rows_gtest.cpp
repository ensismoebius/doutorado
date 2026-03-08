#include <gtest/gtest.h>

#include "MatTestUtils/MatTestUtils.h"
#include "nn/dataLoaders/mat_file_utils.hpp"

using matioCpp::utils::countMatRows;

using namespace nn::dataLoaders;

TEST(CountMatRowsTest, CreatesFixtureAndReadsRowCount)
{
    // Create a tiny MAT fixture (3 rows x 2 cols) with variable name "A"
    const std::string tmpPath = "count_mat_rows_test_fixture.mat";
    const std::string varName = "A";
    const size_t rows = 3;
    const size_t cols = 2;
    double data[6] = {1, 2, 3, 4, 5, 6};

    bool wrote = nn::dataLoaders::test::writeMatDouble(tmpPath, varName, rows, cols, data);
    ASSERT_TRUE(wrote) << "Failed to write MAT fixture: " << tmpPath;

    // Now call the API under test
    auto r = countMatRows(tmpPath, varName);
    EXPECT_EQ(r, rows);

    // Cleanup
    std::remove(tmpPath.c_str());
}
