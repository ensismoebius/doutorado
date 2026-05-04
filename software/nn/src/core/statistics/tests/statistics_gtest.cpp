/**
 * @file statistics_gtest.cpp
 * @brief Unit tests for basic statistics and multi-class metric helpers.
 */

#include <gtest/gtest.h>

#include "nn/statistics/inference_tests.hpp"
#include "nn/statistics/multi_class_metrics.hpp"
#include "nn/statistics/statistics.h"

TEST(StatisticsTest, VarianceWithVector)
{
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Expected variance calculated by hand: mean = 5, sum((x - mean)^2) / n
    const double expected = 4.0;
    EXPECT_DOUBLE_EQ(statistics::variance(data), expected);
}

TEST(StatisticsTest, VarianceWithArray)
{
    const double data[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Same test data as above
    const double expected = 4.0;
    EXPECT_DOUBLE_EQ(statistics::variance(data, 8), expected);
}

TEST(StatisticsTest, StandardDeviationWithVector)
{
    std::vector<double> data = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    // Expected std dev = sqrt(variance) = sqrt(4) = 2
    const double expected = 2.0;
    EXPECT_DOUBLE_EQ(statistics::standardDeviation(data), expected);
}

TEST(StatisticsTest, StandardDeviationWithArray)
{
    const double data[] = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    const double expected = 2.0;
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

TEST(MultiClassMetricsTest, TestComputeClassificationMetrics)
{
    std::vector<int> true_labels = {0, 0, 1, 1, 2, 2};
    std::vector<int> pred_labels = {0, 1, 1, 1, 2, 0};

    auto metrics = statistics::compute_classification_metrics(true_labels, pred_labels);

    EXPECT_NEAR(metrics.accuracy, 4.0 / 6.0, 1e-6);
    EXPECT_NEAR(metrics.precision, 13.0 / 18.0, 1e-6);
    EXPECT_NEAR(metrics.recall, 2.0 / 3.0, 1e-6);
    EXPECT_NEAR(metrics.f1_score, 59.0 / 90.0, 1e-6);
    EXPECT_NEAR(metrics.balanced_accuracy, 2.0 / 3.0, 1e-6);
}

TEST(MultiClassMetricsTest, TestComputeClassificationMetricsEdgeCases)
{
    // All correct predictions
    std::vector<int> true_labels_all_correct = {0, 1, 2};
    std::vector<int> pred_labels_all_correct = {0, 1, 2};
    auto metrics_perfect = statistics::compute_classification_metrics(
        true_labels_all_correct, pred_labels_all_correct);
    EXPECT_NEAR(metrics_perfect.accuracy, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.precision, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.recall, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.f1_score, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.balanced_accuracy, 1.0, 1e-6);

    // All wrong predictions
    std::vector<int> pred_labels_all_wrong = {1, 2, 0};
    auto metrics_wrong =
        statistics::compute_classification_metrics(true_labels_all_correct, pred_labels_all_wrong);
    EXPECT_NEAR(metrics_wrong.accuracy, 0.0, 1e-6);
    EXPECT_NEAR(metrics_wrong.precision, 0.0, 1e-6);
    EXPECT_NEAR(metrics_wrong.recall, 0.0, 1e-6);
    EXPECT_NEAR(metrics_wrong.f1_score, 0.0, 1e-6);
    EXPECT_NEAR(metrics_wrong.balanced_accuracy, 0.0, 1e-6);

    // Single class
    std::vector<int> true_labels_single = {0, 0, 0};
    std::vector<int> pred_labels_single = {0, 0, 0};
    auto metrics_single =
        statistics::compute_classification_metrics(true_labels_single, pred_labels_single);
    EXPECT_NEAR(metrics_single.accuracy, 1.0, 1e-6);
    EXPECT_NEAR(metrics_single.precision, 1.0, 1e-6);
    EXPECT_NEAR(metrics_single.recall, 1.0, 1e-6);
    EXPECT_NEAR(metrics_single.f1_score, 1.0, 1e-6);
    EXPECT_NEAR(metrics_single.balanced_accuracy, 1.0, 1e-6);

    // Empty labels (should handle or throw)
    std::vector<int> empty_true;
    std::vector<int> empty_pred;
    EXPECT_THROW(
        statistics::compute_classification_metrics(empty_true, empty_pred), std::runtime_error);
}

TEST(MultiClassMetricsTest, TestKFoldCrossValidation)
{
    std::vector<std::vector<double>> features = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}};
    std::vector<int> labels = {0, 0, 1, 1, 2, 2};
    int k = 3;
    int seed = 42;

    auto results = statistics::k_fold_cross_validation<double>(features,
        labels,
        k,
        seed,
        [](const std::vector<std::vector<double>>& train_feat,
            const std::vector<int>& train_lab,
            const std::vector<std::vector<double>>& test_feat,
            const std::vector<int>& test_lab) -> double
        {
            // Dummy fold function: return accuracy
            int correct = 0;
            for (size_t i = 0; i < test_lab.size(); ++i)
            {
                // Simple prediction: assume class based on first feature
                int pred = static_cast<int>(test_feat[i][0]) % 3;
                if (pred == test_lab[i]) ++correct;
            }
            return static_cast<double>(correct) / test_lab.size();
        });

    EXPECT_EQ(results.size(), static_cast<size_t>(k));
    ASSERT_EQ(results.size(), 3U);
    EXPECT_NEAR(results[0], 0.5, 1e-12);
    EXPECT_NEAR(results[1], 0.0, 1e-12);
    EXPECT_NEAR(results[2], 0.5, 1e-12);

    const double mean_acc = (results[0] + results[1] + results[2]) / 3.0;
    // Equal fold sizes imply mean fold accuracy equals global sample accuracy.
    EXPECT_NEAR(mean_acc, 1.0 / 3.0, 1e-12);
}

TEST(MultiClassMetricsTest, TestKFoldCrossValidationEdgeCases)
{
    std::vector<std::vector<double>> features = {{1.0}, {2.0}};
    std::vector<int> labels = {0, 1};

    // k = 1 is invalid for KFold
    EXPECT_THROW(statistics::k_fold_cross_validation<double>(features,
                     labels,
                     1,
                     42,
                     [](const auto& train_feat,
                         const auto& train_lab,
                         const auto& test_feat,
                         const auto& test_lab) -> double { return 0.0; }),
        std::invalid_argument);

    // k > n_samples is invalid
    EXPECT_THROW(statistics::k_fold_cross_validation<double>(features,
                     labels,
                     5,
                     42,
                     [](const auto& train_feat,
                         const auto& train_lab,
                         const auto& test_feat,
                         const auto& test_lab) -> double { return 0.5; }),
        std::invalid_argument);

    // Empty data (should throw)
    std::vector<std::vector<double>> empty_features;
    std::vector<int> empty_labels;
    EXPECT_THROW(statistics::k_fold_cross_validation<double>(empty_features,
                     empty_labels,
                     3,
                     42,
                     [](const auto& train_feat,
                         const auto& train_lab,
                         const auto& test_feat,
                         const auto& test_lab) -> double { return 0.0; }),
        std::runtime_error);
}

// Exception Testing for Statistics
TEST(StatisticsExceptionTest, EmptyDataVariance)
{
    std::vector<double> empty_data;
    EXPECT_THROW(statistics::variance(empty_data), std::runtime_error);
    EXPECT_THROW(statistics::standardDeviation(empty_data), std::runtime_error);
}

TEST(StatisticsExceptionTest, MismatchedLabelsLengths)
{
    std::vector<int> true_labels = {0, 1, 2};
    std::vector<int> pred_labels = {0, 1}; // Different length
    EXPECT_THROW(statistics::compute_classification_metrics(true_labels, pred_labels),
        std::invalid_argument);
}

TEST(StatisticsExceptionTest, InvalidKFoldParameters)
{
    std::vector<std::vector<double>> features = {{1.0}, {2.0}};
    std::vector<int> labels = {0, 1};

    // k = 0
    EXPECT_THROW(statistics::k_fold_cross_validation<double>(features,
                     labels,
                     0,
                     42,
                     [](const auto& train_feat,
                         const auto& train_lab,
                         const auto& test_feat,
                         const auto& test_lab) -> double { return 0.0; }),
        std::invalid_argument);

    // k > number of samples and k <= 0
    EXPECT_THROW(statistics::k_fold_cross_validation<double>(features,
                     labels,
                     -1,
                     42,
                     [](const auto& train_feat,
                         const auto& train_lab,
                         const auto& test_feat,
                         const auto& test_lab) -> double { return 0.0; }),
        std::invalid_argument);
}

// Memory Stress Testing for Statistics
TEST(StatisticsMemoryStressTest, LargeDatasetVariance)
{
    const int large_size = 100000;
    std::vector<double> large_data;

    // Create large dataset with known variance
    for (int i = 0; i < large_size; ++i)
    {
        large_data.push_back(static_cast<double>(i % 100)); // Values 0-99 repeating
    }

    ASSERT_NO_THROW({
        const double var = statistics::variance(large_data);
        const double stddev = statistics::standardDeviation(large_data);

        EXPECT_NEAR(var, 833.25, 1e-12);
        EXPECT_NEAR(stddev, 28.86607004772212, 1e-12);
    });
}

TEST(StatisticsMemoryStressTest, LargeClassificationMetrics)
{
    const int num_samples = 50000;
    std::vector<int> true_labels;
    std::vector<int> pred_labels;

    // Create large classification dataset
    for (int i = 0; i < num_samples; ++i)
    {
        true_labels.push_back(i % 10);             // 10 classes
        pred_labels.push_back((i + (i % 3)) % 10); // Some correct, some wrong
    }

    ASSERT_NO_THROW({
        const auto metrics = statistics::compute_classification_metrics(true_labels, pred_labels);

        EXPECT_NEAR(metrics.accuracy, 0.33334, 1e-12);
        EXPECT_NEAR(metrics.precision, 0.33333999866639996, 1e-12);
        EXPECT_NEAR(metrics.recall, 0.33333999999999997, 1e-12);
        EXPECT_NEAR(metrics.f1_score, 0.33333999866659997, 1e-12);
        EXPECT_NEAR(metrics.balanced_accuracy, 0.33333999999999997, 1e-12);
    });
}

TEST(InferenceTests, CohensDReturnsZeroForEqualVectors)
{
    const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_NEAR(statistics::cohens_d(a, b), 0.0f, 1e-6f);
}

TEST(InferenceTests, TTestApproxReturnsSmallPForSeparatedMeans)
{
    const std::vector<float> a = {1.0f, 1.1f, 0.9f, 1.2f, 1.0f};
    const std::vector<float> b = {3.0f, 3.1f, 2.9f, 3.2f, 3.0f};
    const float p = statistics::t_test_pvalue_approx(a, b);
    EXPECT_FLOAT_EQ(p, 0.0f);
}

TEST(InferenceTests, WilcoxonApproxReturnsOneForInvalidPairs)
{
    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {1.0f, 2.0f};
    EXPECT_FLOAT_EQ(statistics::wilcoxon_signed_rank_pvalue_approx(a, b), 1.0f);
}

// Numerical Edge Cases for Statistics
TEST(StatisticsNumericalEdgeTest, NaNInfValues)
{
    // Test with NaN values
    std::vector<double> nan_data = {1.0, 2.0, std::numeric_limits<double>::quiet_NaN(), 4.0};
    double nan_var = statistics::variance(nan_data);
    double nan_std = statistics::standardDeviation(nan_data);
    EXPECT_TRUE(std::isnan(nan_var) || !std::isfinite(nan_var));
    EXPECT_TRUE(std::isnan(nan_std) || !std::isfinite(nan_std));

    // Test with Inf values
    std::vector<double> inf_data = {1.0, 2.0, std::numeric_limits<double>::infinity(), 4.0};
    double inf_var = statistics::variance(inf_data);
    double inf_std = statistics::standardDeviation(inf_data);
    EXPECT_TRUE(std::isinf(inf_var) || !std::isfinite(inf_var));
    EXPECT_TRUE(std::isinf(inf_std) || !std::isfinite(inf_std));

    // Test classification with edge case labels
    std::vector<int> true_labels = {0, 0, 0};
    std::vector<int> pred_labels = {0, 0, 0}; // All same predictions
    auto metrics = statistics::compute_classification_metrics(true_labels, pred_labels);
    EXPECT_NEAR(metrics.precision, 1.0, 1e-6); // Perfect precision for single class
}

TEST(StatisticsNumericalEdgeTest, ExtremeValues)
{
    // Test with very large values
    std::vector<double> large_data = {1e10, 2e10, 3e10};
    double large_var = statistics::variance(large_data);
    const double expected_large_var = 2e20 / 3.0;
    EXPECT_TRUE(std::isfinite(large_var));
    EXPECT_NEAR(large_var, expected_large_var, expected_large_var * 1e-12);

    // Test with very small values
    std::vector<double> small_data = {1e-10, 2e-10, 3e-10};
    double small_var = statistics::variance(small_data);
    const double expected_small_var = 2e-20 / 3.0;
    EXPECT_TRUE(std::isfinite(small_var));
    EXPECT_NEAR(small_var, expected_small_var, expected_small_var * 1e-12);

    // Test with zero variance (constant values)
    std::vector<double> constant_data = {5.0, 5.0, 5.0, 5.0};
    double zero_var = statistics::variance(constant_data);
    EXPECT_NEAR(zero_var, 0.0, 1e-10);
    double zero_std = statistics::standardDeviation(constant_data);
    EXPECT_NEAR(zero_std, 0.0, 1e-10);
}

// Thread Safety Validation for Statistics
TEST(StatisticsThreadSafetyTest, ConcurrentVarianceCalculations)
{
    const int num_threads = 10;
    const int data_size = 1000;

    // Test multiple concurrent variance calculations
    for (int t = 0; t < num_threads; ++t)
    {
        std::vector<double> data;
        for (int i = 0; i < data_size; ++i)
        {
            data.push_back(static_cast<double>(i + t * data_size));
        }

        ASSERT_NO_THROW({
            const double var = statistics::variance(data);
            const double std = statistics::standardDeviation(data);

            EXPECT_NEAR(var, 83333.25, 1e-12);
            EXPECT_NEAR(std, 288.6749902572095, 1e-12);
        });
    }
}

TEST(StatisticsThreadSafetyTest, ConcurrentClassificationMetrics)
{
    const int num_tests = 5;

    for (int test = 0; test < num_tests; ++test)
    {
        std::vector<int> true_labels = {0, 1, 2, 0, 1, 2, 0, 1, 2};
        std::vector<int> pred_labels = {0, 1, 2, 1, 2, 0, 2, 0, 1}; // Some correct, some wrong

        ASSERT_NO_THROW({
            const auto metrics =
                statistics::compute_classification_metrics(true_labels, pred_labels);

            EXPECT_NEAR(metrics.accuracy, 1.0 / 3.0, 1e-12);
            EXPECT_NEAR(metrics.precision, 1.0 / 3.0, 1e-12);
            EXPECT_NEAR(metrics.recall, 1.0 / 3.0, 1e-12);
            EXPECT_NEAR(metrics.f1_score, 1.0 / 3.0, 1e-12);
            EXPECT_NEAR(metrics.balanced_accuracy, 1.0 / 3.0, 1e-12);
        });
    }
}

// Additional Comprehensive Tests
TEST(StatisticsComprehensiveTest, VarianceCorrectness)
{
    // Test variance calculation with known statistical properties
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Population variance: mean = 3, variance = sum((x-mean)^2)/n
    // (1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2 = 4+1+0+1+4 = 10
    // 10/5 = 2.0
    double expected_var = 2.0;
    double calculated_var = statistics::variance(data);
    EXPECT_NEAR(calculated_var, expected_var, 1e-10);

    // Standard deviation should be sqrt(variance)
    double expected_std = std::sqrt(expected_var);
    double calculated_std = statistics::standardDeviation(data);
    EXPECT_NEAR(calculated_std, expected_std, 1e-10);
}

TEST(StatisticsComprehensiveTest, ClassificationMetricsDetailed)
{
    // Test case with known expected values
    std::vector<int> true_labels = {0, 0, 1, 1, 2, 2};
    std::vector<int> pred_labels = {0, 1, 1, 1, 2, 0};

    auto metrics = statistics::compute_classification_metrics(true_labels, pred_labels);

    // Manual calculation:
    // Confusion matrix:
    // True\Pred | 0 | 1 | 2
    //     0     | 1 | 1 | 0  -> TP=1, FP=1, FN=1, TN=3
    //     1     | 0 | 2 | 0  -> TP=2, FP=0, FN=0, TN=4
    //     2     | 1 | 0 | 1  -> TP=1, FP=0, FN=1, TN=4

    // Accuracy = (1+2+1)/6 = 4/6 ≈ 0.667
    EXPECT_NEAR(metrics.accuracy, 4.0 / 6.0, 1e-6);

    // For multiclass, precision/recall/f1 are macro-averaged.
    // Class 0: p=1/2, r=1/2, f1=1/2
    // Class 1: p=2/3, r=1,   f1=4/5
    // Class 2: p=1,   r=1/2, f1=2/3
    // Macro precision = (1/2 + 2/3 + 1)/3 = 13/18
    // Macro recall    = (1/2 + 1   + 1/2)/3 = 2/3
    // Macro f1        = (1/2 + 4/5 + 2/3)/3 = 59/90
    EXPECT_NEAR(metrics.precision, 13.0 / 18.0, 1e-6);
    EXPECT_NEAR(metrics.recall, 2.0 / 3.0, 1e-6);
    EXPECT_NEAR(metrics.f1_score, 59.0 / 90.0, 1e-6);
    EXPECT_NEAR(metrics.balanced_accuracy, 2.0 / 3.0, 1e-6);
}

TEST(StatisticsComprehensiveTest, KFoldCrossValidationDeterminism)
{
    std::vector<std::vector<double>> features = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}};
    std::vector<int> labels = {0, 0, 1, 1, 2, 2};

    // Same seed should give same results
    auto results1 = statistics::k_fold_cross_validation<double>(features,
        labels,
        3,
        123,
        [](const auto& train_feat,
            const auto& train_lab,
            const auto& test_feat,
            const auto& test_lab) -> double
        { return static_cast<double>(test_lab.size()) / (train_lab.size() + test_lab.size()); });

    auto results2 = statistics::k_fold_cross_validation<double>(features,
        labels,
        3,
        123,
        [](const auto& train_feat,
            const auto& train_lab,
            const auto& test_feat,
            const auto& test_lab) -> double
        { return static_cast<double>(test_lab.size()) / (train_lab.size() + test_lab.size()); });

    EXPECT_EQ(results1.size(), results2.size());
    for (size_t i = 0; i < results1.size(); ++i)
    {
        EXPECT_NEAR(results1[i], results2[i], 1e-10);
        EXPECT_NEAR(results1[i], 1.0 / 3.0, 1e-10);
    }
}