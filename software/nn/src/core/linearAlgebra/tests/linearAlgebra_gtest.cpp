/**
 * @file linearAlgebra_gtest.cpp
 * @brief Unit tests for small linear algebra helper routines.
 */

#include "gtest/gtest.h"
#include "nn/linearAlgebra/linear_algebra.hpp"
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
    // With a single row, each feature has only one value, so min equals max.
    // The normalized value should be the minimum of the range (0.0).
    std::vector<std::vector<double>> single_row = {{1.0, 2.0, 3.0}};
    linearAlgebra::minMaxNormalizeFeatures(single_row);
    EXPECT_NEAR(single_row[0][0], 0.0, 1e-6);
    EXPECT_NEAR(single_row[0][1], 0.0, 1e-6);
    EXPECT_NEAR(single_row[0][2], 0.0, 1e-6);

    // Single column
    std::vector<std::vector<double>> single_col = {{1.0}, {2.0}, {3.0}};
    linearAlgebra::minMaxNormalizeFeatures(single_col);
    EXPECT_NEAR(single_col[0][0], 0.0, 1e-6);
    EXPECT_NEAR(single_col[1][0], 0.5, 1e-6);
    EXPECT_NEAR(single_col[2][0], 1.0, 1e-6);

    // Constant features (min == max)
    // Should handle division by zero. Current implementation sets to range[0] (0.0).
    std::vector<std::vector<double>> constant = {{2.0, 2.0}, {2.0, 2.0}};
    linearAlgebra::minMaxNormalizeFeatures(constant);
    for (const auto& row : constant)
    {
        for (double val : row)
        {
            EXPECT_NEAR(val, 0.0, 1e-6);
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
TEST(LinearAlgebraTest, TestDotProduct)
{
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_NEAR(linearAlgebra::dotProduct(a, b), 32.0, 1e-6);

    // Test with spans (subranges)
    // a[0..1] = {1,2}, b[1..2] = {5,6}. 1*5 + 2*6 = 5 + 12 = 17
    EXPECT_NEAR(linearAlgebra::dotProduct(std::span(a).subspan(0, 2), std::span(b).subspan(1, 2)),
                17.0,
                1e-6);
}

TEST(LinearAlgebraTest, TestCalcOrthogonalVector)
{
    std::vector<double> v = {1.0, 2.0, 3.0};
    std::vector<double> ortho = linearAlgebra::calcOrthogonalVector(v);
    // Implementation: reverse and flip alternating signs.
    // result[0] = 3.0, result[1] = -2.0, result[2] = 1.0
    EXPECT_EQ(ortho.size(), 3);
    EXPECT_NEAR(ortho[0], 3.0, 1e-6);
    EXPECT_NEAR(ortho[1], -2.0, 1e-6);
    EXPECT_NEAR(ortho[2], 1.0, 1e-6);

    std::vector<double> v4 = {1, 2, 3, 4};
    std::vector<double> o4 = linearAlgebra::calcOrthogonalVector(v4);
    // Dot product check for even dimensions
    double dp = linearAlgebra::dotProduct(v4, o4);
    EXPECT_NEAR(dp, 0.0, 1e-6);
}

TEST(LinearAlgebraTest, TestNormalizeVectorToSum1)
{
    std::vector<double> v = {1.0, 3.0};
    linearAlgebra::normalizeVectorToSum1(v);
    EXPECT_NEAR(v[0], 0.25, 1e-6);
    EXPECT_NEAR(v[1], 0.75, 1e-6);
}

TEST(LinearAlgebraTest, TestConvolution)
{
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0}; // N=4
    std::vector<double> kernel = {1.0, 0.0, -1.0};   // K=3
    // Expected result: {1, 2, 2, 2}

    linearAlgebra::convolution(data, kernel);
    EXPECT_NEAR(data[0], 1.0, 1e-6);
    EXPECT_NEAR(data[1], 2.0, 1e-6);
    EXPECT_NEAR(data[2], 2.0, 1e-6);
    EXPECT_NEAR(data[3], 2.0, 1e-6);
}

TEST(LinearAlgebraTest, TestDerivativeLevels)
{
    std::vector<double> v = {1.0, 4.0, 9.0, 16.0};
    auto d1 = linearAlgebra::derivative(v, 1);
    ASSERT_EQ(d1.size(), 3U);
    EXPECT_NEAR(d1[0], 3.0, 1e-6);
    EXPECT_NEAR(d1[1], 5.0, 1e-6);
    EXPECT_NEAR(d1[2], 7.0, 1e-6);

    std::vector<double> empty;
    auto d_empty = linearAlgebra::derivative(empty, 2);
    ASSERT_EQ(d_empty.size(), 1U);
    EXPECT_NEAR(d_empty[0], 0.0, 1e-9);
}

TEST(LinearAlgebraTest, TestNormalizeVectorToRangeAndExceptions)
{
    std::vector<double> v = {2.0, 4.0, 6.0};
    linearAlgebra::normalizeVectorToRange(v, -1.0, 1.0);
    EXPECT_NEAR(v.front(), -1.0, 1e-6);
    EXPECT_NEAR(v.back(), 1.0, 1e-6);

    std::vector<double> same = {3.0, 3.0, 3.0};
    linearAlgebra::normalizeVectorToRange(same, 0.0, 1.0);
    for (double x : same)
    {
        EXPECT_NEAR(x, 0.0, 1e-9);
    }

    EXPECT_THROW((void) linearAlgebra::normalizeVectorToRange(v, 2.0, 2.0), std::runtime_error);
}

TEST(LinearAlgebraTest, TestNormalizeAllPositiveAndDotException)
{
    std::vector<double> v = {-2.0, 0.0, 2.0};
    linearAlgebra::normalizeVectorToSum1AllPositive(v);
    for (double x : v)
    {
        EXPECT_GT(x, 0.0);
    }
    EXPECT_NEAR(v[0] + v[1] + v[2], 1.0, 1e-9);

    std::vector<double> a = {1.0, 2.0};
    std::vector<double> b = {1.0};
    EXPECT_THROW((void) linearAlgebra::dotProduct(a, b), std::invalid_argument);
}

TEST(LinearAlgebraTest, TestConvolutionEdgeCasesAndDct)
{
    std::vector<double> data_empty;
    std::vector<double> kernel = {1.0, -1.0};
    EXPECT_FALSE(linearAlgebra::convolution(data_empty, kernel));

    std::vector<double> data = {1.0, 2.0, 3.0};
    std::vector<double> kernel_empty;
    EXPECT_FALSE(linearAlgebra::convolution(data, kernel_empty));

    std::vector<double> dct_in = {1.0, 1.0, 1.0, 1.0};
    linearAlgebra::discreteCosineTransform(dct_in);
    EXPECT_GT(std::abs(dct_in[0]), 0.0);
}

TEST(LinearAlgebraTest, TestScaleSolveAndResizeCentered)
{
    std::vector<std::vector<double>> matrix = {
        {2.0, 1.0, 5.0},
        {4.0, 4.0, 16.0},
    };
    linearAlgebra::scaleMatrix(matrix);
    auto solution = linearAlgebra::solveMatrix(matrix);
    ASSERT_EQ(solution.size(), 2U);
    EXPECT_NEAR(solution[0], 1.0, 1e-6);
    EXPECT_NEAR(solution[1], 3.0, 1e-6);

    std::vector<double> centered = {1, 2, 3, 4, 5};
    linearAlgebra::resizeCentered(centered, 9, 0.0);
    ASSERT_EQ(centered.size(), 9U);
    EXPECT_NEAR(centered[2], 1.0, 1e-9);
    EXPECT_NEAR(centered[6], 5.0, 1e-9);

    linearAlgebra::resizeCentered(centered, 3, 0.0);
    ASSERT_EQ(centered.size(), 3U);
    EXPECT_NEAR(centered[0], 2.0, 1e-9);
    EXPECT_NEAR(centered[1], 3.0, 1e-9);
    EXPECT_NEAR(centered[2], 4.0, 1e-9);
}
