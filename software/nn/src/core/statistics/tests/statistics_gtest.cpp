#include <gtest/gtest.h>

#include "multiClassMetrics.h"
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

TEST(MultiClassMetricsTest, TestComputeClassificationMetrics)
{
    std::vector<int> true_labels = {0, 0, 1, 1, 2, 2};
    std::vector<int> pred_labels = {0, 1, 1, 1, 2, 0};

    auto metrics = statistics::compute_classification_metrics(true_labels, pred_labels);

    EXPECT_GE(metrics.accuracy, 0.0);
    EXPECT_LE(metrics.accuracy, 1.0);
    EXPECT_GE(metrics.precision, 0.0);
    EXPECT_LE(metrics.precision, 1.0);
    EXPECT_GE(metrics.recall, 0.0);
    EXPECT_LE(metrics.recall, 1.0);
    EXPECT_GE(metrics.f1_score, 0.0);
    EXPECT_LE(metrics.f1_score, 1.0);
    EXPECT_GE(metrics.balanced_accuracy, 0.0);
    EXPECT_LE(metrics.balanced_accuracy, 1.0);
}

TEST(MultiClassMetricsTest, TestComputeClassificationMetricsEdgeCases)
{
    // All correct predictions
    std::vector<int> true_labels_all_correct = {0, 1, 2};
    std::vector<int> pred_labels_all_correct = {0, 1, 2};
    auto metrics_perfect = statistics::compute_classification_metrics(true_labels_all_correct,
                                                                      pred_labels_all_correct);
    EXPECT_NEAR(metrics_perfect.accuracy, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.precision, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.recall, 1.0, 1e-6);
    EXPECT_NEAR(metrics_perfect.f1_score, 1.0, 1e-6);

    // All wrong predictions
    std::vector<int> pred_labels_all_wrong = {1, 2, 0};
    auto metrics_wrong =
        statistics::compute_classification_metrics(true_labels_all_correct, pred_labels_all_wrong);
    EXPECT_NEAR(metrics_wrong.accuracy, 0.0, 1e-6);

    // Single class
    std::vector<int> true_labels_single = {0, 0, 0};
    std::vector<int> pred_labels_single = {0, 0, 0};
    auto metrics_single =
        statistics::compute_classification_metrics(true_labels_single, pred_labels_single);
    EXPECT_NEAR(metrics_single.accuracy, 1.0, 1e-6);

    // Empty labels (should handle or throw)
    std::vector<int> empty_true;
    std::vector<int> empty_pred;
    EXPECT_THROW(statistics::compute_classification_metrics(empty_true, empty_pred),
                 std::runtime_error);
}

TEST(MultiClassMetricsTest, TestKFoldCrossValidation)
{
    std::vector<std::vector<double>> features = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}};
    std::vector<int> labels = {0, 0, 1, 1, 2, 2};
    int k = 3;
    int seed = 42;

    auto results = statistics::k_fold_cross_validation<double>(
        features,
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
    for (double acc : results)
    {
        EXPECT_GE(acc, 0.0);
        EXPECT_LE(acc, 1.0);
    }
}

TEST(MultiClassMetricsTest, TestKFoldCrossValidationEdgeCases)
{
    std::vector<std::vector<double>> features = {{1.0}, {2.0}};
    std::vector<int> labels = {0, 1};

    // k = 1 (leave-one-out like)
    auto results_k1 = statistics::k_fold_cross_validation<double>(
        features,
        labels,
        1,
        42,
        [](const auto& train_feat,
           const auto& train_lab,
           const auto& test_feat,
           const auto& test_lab) -> double
        {
            return test_lab[0] == 0 ? 1.0 : 0.0; // Dummy
        });
    EXPECT_EQ(results_k1.size(), 1U);

    // k > n_samples (should handle)
    auto results_k_large = statistics::k_fold_cross_validation<double>(
        features,
        labels,
        5,
        42,
        [](const auto& train_feat,
           const auto& train_lab,
           const auto& test_feat,
           const auto& test_lab) -> double { return 0.5; });
    EXPECT_EQ(results_k_large.size(), 5U);

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
                                                                const auto& test_lab) -> double
                                                             { return 0.0; }),
                 std::runtime_error);
}