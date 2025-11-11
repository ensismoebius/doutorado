#include <gtest/gtest.h>

#include "statistics.h"

TEST(StatisticsTest, VarianceWithVector)
{
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Expected variance calculated by hand: mean = 5, sum((x - mean)^2) / n
    double expected = 4.0;
    EXPECT_DOUBLE_EQ(statistics::variance(data), expected);
}

TEST(StatisticsTest, VarianceWithArray)
{
    double data[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Same test data as above
    double expected = 4.0;
    EXPECT_DOUBLE_EQ(statistics::variance(data, 8), expected);
}

TEST(StatisticsTest, StandardDeviationWithVector)
{
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Expected std dev = sqrt(variance) = sqrt(4) = 2
    double expected = 2.0;
    EXPECT_DOUBLE_EQ(statistics::standardDeviation(data), expected);
}

TEST(StatisticsTest, StandardDeviationWithArray)
{
    double data[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    double expected = 2.0;
    EXPECT_DOUBLE_EQ(statistics::standardDeviation(data, 8), expected);
}

TEST(StatisticsTest, VarianceWithSingleValue)
{
    std::vector<double> data = {5.0};
    EXPECT_DOUBLE_EQ(statistics::variance(data), 0.0);
}

TEST(StatisticsTest, StandardDeviationWithSingleValue)
{
    std::vector<double> data = {5.0};
    EXPECT_DOUBLE_EQ(statistics::standardDeviation(data), 0.0);
}