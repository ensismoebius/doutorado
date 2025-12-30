#include "../linearAlgebra.h"
#include "gtest/gtest.h"

// Define a simple test fixture if needed, or just use TEST
TEST(LinearAlgebraTest, BasicAssertion)
{
    // Example: Check if 1 + 1 equals 2
    ASSERT_EQ(1 + 1, 2);
}

TEST(LinearAlgebraTest, TestMinMaxNormalizeFeatures)
{
    std::vector<std::vector<double>> features = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
    std::vector<double> range = {0.0, 1.0};

    linearAlgebra::minMaxNormalizeFeatures(features, range);

    // Check that all values are in [0, 1]
    for (const auto& row : features)
    {
        for (double val : row)
        {
            EXPECT_GE(val, 0.0);
            EXPECT_LE(val, 1.0);
        }
    }

    // For first feature (column 0): min=1, max=7, normalized: (1-1)/(7-1)=0, (4-1)/(7-1)=0.5,
    // (7-1)/(7-1)=1
    EXPECT_NEAR(features[0][0], 0.0, 1e-6);
    EXPECT_NEAR(features[1][0], 0.5, 1e-6);
    EXPECT_NEAR(features[2][0], 1.0, 1e-6);
}

TEST(LinearAlgebraTest, TestMinMaxNormalizeFeaturesEdgeCases)
{
    // Single row
    std::vector<std::vector<double>> single_row = {{1.0, 2.0, 3.0}};
    linearAlgebra::minMaxNormalizeFeatures(single_row);
    EXPECT_NEAR(single_row[0][0], 0.0, 1e-6);
    EXPECT_NEAR(single_row[0][1], 0.5, 1e-6);
    EXPECT_NEAR(single_row[0][2], 1.0, 1e-6);

    // Single column
    std::vector<std::vector<double>> single_col = {{1.0}, {2.0}, {3.0}};
    linearAlgebra::minMaxNormalizeFeatures(single_col);
    EXPECT_NEAR(single_col[0][0], 0.0, 1e-6);
    EXPECT_NEAR(single_col[1][0], 0.5, 1e-6);
    EXPECT_NEAR(single_col[2][0], 1.0, 1e-6);

    // Constant features (min == max)
    std::vector<std::vector<double>> constant = {{2.0, 2.0}, {2.0, 2.0}};
    linearAlgebra::minMaxNormalizeFeatures(constant);
    // Should handle division by zero, perhaps set to 0.5 or leave as is
    for (const auto& row : constant)
    {
        for (double val : row)
        {
            EXPECT_TRUE(std::isnan(val) || val == 0.5 || val == 2.0); // Depending on implementation
        }
    }

    // Empty matrix (should handle)
    std::vector<std::vector<double>> empty;
    linearAlgebra::minMaxNormalizeFeatures(empty);
    EXPECT_TRUE(empty.empty());

    // Custom range
    std::vector<std::vector<double>> custom = {{0.0}, {1.0}};
    linearAlgebra::minMaxNormalizeFeatures(custom, {-1.0, 1.0});
    EXPECT_NEAR(custom[0][0], -1.0, 1e-6);
    EXPECT_NEAR(custom[1][0], 1.0, 1e-6);
}

// You can add more test cases here
// TEST(LinearAlgebraTest, AnotherTest) {
//     // Your test code here
// }
