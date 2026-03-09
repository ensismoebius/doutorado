#include "gtest/gtest.h"

#include "../Experiment02Reporting.hpp"

TEST(Experiment02ReportingTest, AggregateFoldResultsHandlesEmptyInput)
{
    const std::vector<FoldResult> empty_results;
    const auto aggregated = aggregate_fold_results(empty_results);

    EXPECT_DOUBLE_EQ(aggregated.classification.accuracy, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.classification.precision, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.classification.recall, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.classification.f1_score, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.classification.mcc, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.alpha, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.beta, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.G1, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.G2, 0.0);
    EXPECT_DOUBLE_EQ(aggregated.total_time_sec, 0.0);
}

TEST(Experiment02ReportingTest, AggregateFoldResultsComputesAverages)
{
    FoldResult fold_a;
    fold_a.metrics.accuracy = 0.6;
    fold_a.metrics.precision = 0.5;
    fold_a.metrics.recall = 0.4;
    fold_a.metrics.f1_score = 0.45;
    fold_a.metrics.mcc = 0.2;
    fold_a.para_metrics.alpha = 0.1;
    fold_a.para_metrics.beta = 0.2;
    fold_a.para_metrics.G1 = 0.3;
    fold_a.para_metrics.G2 = 0.4;
    fold_a.fold_time_sec = 2.0;

    FoldResult fold_b;
    fold_b.metrics.accuracy = 0.8;
    fold_b.metrics.precision = 0.7;
    fold_b.metrics.recall = 0.6;
    fold_b.metrics.f1_score = 0.65;
    fold_b.metrics.mcc = 0.4;
    fold_b.para_metrics.alpha = 0.5;
    fold_b.para_metrics.beta = 0.6;
    fold_b.para_metrics.G1 = 0.7;
    fold_b.para_metrics.G2 = 0.8;
    fold_b.fold_time_sec = 4.0;

    const std::vector<FoldResult> folds{fold_a, fold_b};
    const auto aggregated = aggregate_fold_results(folds);

    EXPECT_DOUBLE_EQ(aggregated.classification.accuracy, 0.7);
    EXPECT_DOUBLE_EQ(aggregated.classification.precision, 0.6);
    EXPECT_DOUBLE_EQ(aggregated.classification.recall, 0.5);
    EXPECT_DOUBLE_EQ(aggregated.classification.f1_score, 0.55);
    EXPECT_DOUBLE_EQ(aggregated.classification.mcc, 0.3);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.alpha, 0.3);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.beta, 0.4);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.G1, 0.5);
    EXPECT_DOUBLE_EQ(aggregated.paraconsistent.G2, 0.6);
    EXPECT_DOUBLE_EQ(aggregated.total_time_sec, 6.0);
}
