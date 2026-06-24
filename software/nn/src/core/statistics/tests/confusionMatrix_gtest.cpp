/**
 * @file confusionMatrix_gtest.cpp
 * @brief Unit tests for confusion-matrix derived metrics.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "statistics/confusion_matrix.hpp"

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

TEST(ConfusionMatrixTest, FalseNegativeRateAndTruePositiveRateNumeric)
{
    // FN rate = fn/(fn+tp), TP rate = tp/(fn+tp)
    EXPECT_DOUBLE_EQ(statistics::falseNegativeRate(8, 2), 0.2);
    EXPECT_DOUBLE_EQ(statistics::truePositiveRate(8, 2), 0.8);
}

TEST(ConfusionMatrixTest, AccuracyPrecisionRecallNumeric)
{
    // tp=8, tn=9, fp=1, fn=2
    EXPECT_DOUBLE_EQ(statistics::accuracyRate(8, 9, 1, 2), 17.0 / 20.0);
    EXPECT_DOUBLE_EQ(statistics::precision(8, 1), 8.0 / 9.0);
    EXPECT_DOUBLE_EQ(statistics::recall(8, 2), 0.8);
}

TEST(ConfusionMatrixTest, MatrixOverloadsMatchNumericOverloads)
{
    statistics::ConfusionMatrix matrix{};
    matrix.truePositive = 8;
    matrix.falsePositive = 1;
    matrix.falseNegative = 2;
    matrix.trueNegative = 9;

    EXPECT_DOUBLE_EQ(statistics::falseNegativeRate(matrix),
        statistics::falseNegativeRate(matrix.truePositive, matrix.falseNegative));
    EXPECT_DOUBLE_EQ(statistics::accuracyRate(matrix),
        statistics::accuracyRate(
            matrix.truePositive, matrix.trueNegative, matrix.falsePositive, matrix.falseNegative));
    EXPECT_DOUBLE_EQ(statistics::precision(matrix),
        statistics::precision(matrix.truePositive, matrix.falsePositive));
    EXPECT_DOUBLE_EQ(
        statistics::recall(matrix), statistics::recall(matrix.truePositive, matrix.falseNegative));

    // TPR(matrix) must use (truePositive, falseNegative) = tp/(tp+fn) (audit M-2).
    EXPECT_DOUBLE_EQ(statistics::truePositiveRate(matrix),
        statistics::truePositiveRate(matrix.truePositive, matrix.falseNegative));
    EXPECT_DOUBLE_EQ(statistics::truePositiveRate(matrix), 8.0 / 10.0);
}

TEST(ConfusionMatrixTest, CalculateEERFromRates)
{
    std::vector<double> fpr = {0.1, 0.2, 0.3, 0.4};
    std::vector<double> fnr = {0.4, 0.3, 0.2, 0.1};
    double eer = 0.0;

    statistics::calculateEER(eer, fpr, fnr);

    // For symmetric crossing around x=y, EER should be between 0 and 1.
    EXPECT_GE(eer, 0.0);
    EXPECT_LE(eer, 1.0);
}

TEST(ConfusionMatrixTest, CalculateEERFromConfusionMatricesBuildsRateVectors)
{
    std::vector<statistics::ConfusionMatrix> cms = {
        {90, 10, 40, 60}, // fpr=10/(10+60)=0.142857..., fnr=40/(40+90)=0.307692...
        {80, 20, 30, 70},
        {70, 30, 20, 80},
        {60, 40, 10, 90},
    };

    double eer = 0.0;
    std::vector<double> fpr;
    std::vector<double> fnr;

    statistics::calculateEER(cms, eer, fpr, fnr);

    ASSERT_EQ(fpr.size(), cms.size());
    ASSERT_EQ(fnr.size(), cms.size());

    for (double value : fpr)
    {
        EXPECT_GE(value, 0.0);
        EXPECT_LE(value, 1.0);
    }
    for (double value : fnr)
    {
        EXPECT_GE(value, 0.0);
        EXPECT_LE(value, 1.0);
    }
    EXPECT_TRUE(std::isfinite(eer));
}