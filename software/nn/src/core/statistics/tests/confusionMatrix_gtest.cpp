#include <gtest/gtest.h>

#include "nn/statistics/confusion_matrix.hpp"

TEST(ConfusionMatrixTest, FalsePositiveRate)
{
    statistics::ConfusionMatrix matrix;
    matrix.falsePositive = 10;
    matrix.trueNegative = 90;

    // FPR = FP / (FP + TN) = 10 / (10 + 90) = 0.1
    double expected = 0.1;
    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(matrix), expected);
    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(10, 90), expected);
}

TEST(ConfusionMatrixTest, FalsePositiveRateWithZeroDenominator)
{
    statistics::ConfusionMatrix matrix;
    matrix.falsePositive = 0;
    matrix.trueNegative = 0;

    // When denominator is zero, the function should handle it gracefully
    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(matrix), 0.0);
    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(0, 0), 0.0);
}

TEST(ConfusionMatrixTest, FalsePositiveRateWithZeroFP)
{
    statistics::ConfusionMatrix matrix;
    matrix.falsePositive = 0;
    matrix.trueNegative = 100;

    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(matrix), 0.0);
    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(0, 100), 0.0);
}

TEST(ConfusionMatrixTest, FalsePositiveRateWithZeroTN)
{
    statistics::ConfusionMatrix matrix;
    matrix.falsePositive = 50;
    matrix.trueNegative = 0;

    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(matrix), 1.0);
    EXPECT_DOUBLE_EQ(statistics::falsePositiveRate(50, 0), 1.0);
}