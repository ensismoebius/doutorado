/**
 * @file kfold_gtest.cpp
 * @brief Unit tests for KFold and StratifiedKFold utilities.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "statistics/kfold.hpp"
#include "statistics/multi_class_metrics.hpp"

namespace
{

auto to_set(const std::vector<std::size_t>& values) -> std::set<std::size_t>
{
    return std::set<std::size_t>(values.begin(), values.end());
}

} // namespace

TEST(KFoldTest, CoversAllSamplesWithoutOverlap)
{
    statistics::KFold splitter(/*n_splits=*/3, /*shuffle=*/false);
    const auto folds = splitter.split(/*n_samples=*/10);

    ASSERT_EQ(folds.size(), 3U);

    std::set<std::size_t> union_test;
    for (const auto& fold : folds)
    {
        const auto train_set = to_set(fold.train_indices);
        const auto test_set = to_set(fold.test_indices);

        EXPECT_EQ(fold.train_indices.size() + fold.test_indices.size(), 10U);

        // Train and test partitions in one fold must be disjoint.
        for (const auto idx : test_set)
        {
            EXPECT_EQ(train_set.count(idx), 0U);
            union_test.insert(idx);
        }
    }

    EXPECT_EQ(union_test.size(), 10U);
    for (std::size_t i = 0; i < 10U; ++i)
    {
        EXPECT_EQ(union_test.count(i), 1U);
    }
}

TEST(KFoldTest, ShuffleIsDeterministicBySeed)
{
    statistics::KFold splitter_a(/*n_splits=*/4, /*shuffle=*/true, /*seed=*/42U);
    statistics::KFold splitter_b(/*n_splits=*/4, /*shuffle=*/true, /*seed=*/42U);

    const auto folds_a = splitter_a.split(20U);
    const auto folds_b = splitter_b.split(20U);

    ASSERT_EQ(folds_a.size(), folds_b.size());
    for (std::size_t i = 0; i < folds_a.size(); ++i)
    {
        EXPECT_EQ(folds_a[i].test_indices, folds_b[i].test_indices);
        EXPECT_EQ(folds_a[i].train_indices, folds_b[i].train_indices);
    }
}

TEST(KFoldTest, InvalidParametersThrow)
{
    EXPECT_THROW(
        {
            const auto folds = statistics::KFold(1U, false).split(10U);
            (void) folds;
        },
        std::invalid_argument);
    EXPECT_THROW(
        {
            const auto folds = statistics::KFold(5U, false).split(4U);
            (void) folds;
        },
        std::invalid_argument);
}

TEST(StratifiedKFoldTest, PreservesClassDistributionAcrossFolds)
{
    // 15 labels: class 0, 1, 2 each appears exactly 5 times.
    std::vector<int> labels = {
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
    };

    statistics::StratifiedKFold splitter(/*n_splits=*/5, /*shuffle=*/false);
    const auto folds = splitter.split(labels);

    ASSERT_EQ(folds.size(), 5U);

    for (const auto& fold : folds)
    {
        std::array<int, 3> class_counts = {0, 0, 0};
        for (const std::size_t idx : fold.test_indices)
        {
            ++class_counts[static_cast<std::size_t>(labels[idx])];
        }

        // With 5 folds and 5 examples per class, each fold gets 1 sample per class.
        EXPECT_EQ(class_counts[0], 1);
        EXPECT_EQ(class_counts[1], 1);
        EXPECT_EQ(class_counts[2], 1);
    }
}

TEST(StratifiedKFoldTest, InvalidParametersThrow)
{
    std::vector<int> labels = {0, 1, 0, 1};
    EXPECT_THROW(
        {
            const auto folds = statistics::StratifiedKFold(1U, false).split(labels);
            (void) folds;
        },
        std::invalid_argument);
    EXPECT_THROW(
        {
            const auto folds = statistics::StratifiedKFold(5U, false).split(labels);
            (void) folds;
        },
        std::invalid_argument);
}

TEST(KFoldCompatibilityTest, ExistingKFoldCrossValidationStillWorks)
{
    std::vector<std::vector<double>> features = {
        {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}, {5.0, 6.0}, {6.0, 7.0}};
    std::vector<int> labels = {0, 0, 1, 1, 2, 2};

    const auto results = statistics::k_fold_cross_validation<double>(features,
        labels,
        /*k=*/3,
        /*seed=*/42,
        [](const std::vector<std::vector<double>>& train_feat,
            const std::vector<int>& train_lab,
            const std::vector<std::vector<double>>& test_feat,
            const std::vector<int>& test_lab) -> double
        {
            EXPECT_FALSE(train_feat.empty());
            EXPECT_FALSE(train_lab.empty());
            EXPECT_FALSE(test_feat.empty());
            EXPECT_FALSE(test_lab.empty());
            return static_cast<double>(test_lab.size());
        });

    ASSERT_EQ(results.size(), 3U);
}

TEST(StratifiedKFoldTest, ShuffleIsDeterministicBySeed)
{
    std::vector<int> labels = {
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        3,
    };

    statistics::StratifiedKFold a(/*n_splits=*/3, /*shuffle=*/true, /*seed=*/99U);
    statistics::StratifiedKFold b(/*n_splits=*/3, /*shuffle=*/true, /*seed=*/99U);

    const auto folds_a = a.split(labels);
    const auto folds_b = b.split(labels);
    ASSERT_EQ(folds_a.size(), folds_b.size());
    for (std::size_t i = 0; i < folds_a.size(); ++i)
    {
        EXPECT_EQ(folds_a[i].test_indices, folds_b[i].test_indices);
        EXPECT_EQ(folds_a[i].train_indices, folds_b[i].train_indices);
    }
}

TEST(NestedKFoldTest, ProducesConsistentOuterAndInnerSplits)
{
    statistics::NestedKFold splitter(/*outer=*/3, /*inner=*/2, /*shuffle=*/true, /*seed=*/7U);
    const auto nested = splitter.split(/*n_samples=*/12U);

    ASSERT_EQ(nested.size(), 3U);
    for (const auto& outer : nested)
    {
        EXPECT_FALSE(outer.test_indices.empty());
        ASSERT_EQ(outer.inner_splits.size(), 2U);

        std::set<std::size_t> outer_test(outer.test_indices.begin(), outer.test_indices.end());
        for (const auto& inner : outer.inner_splits)
        {
            // Inner splits are drawn from outer train set only.
            for (std::size_t idx : inner.train_indices)
            {
                EXPECT_EQ(outer_test.count(idx), 0U);
            }
            for (std::size_t idx : inner.test_indices)
            {
                EXPECT_EQ(outer_test.count(idx), 0U);
            }
            EXPECT_FALSE(inner.train_indices.empty());
            EXPECT_FALSE(inner.test_indices.empty());
        }
    }
}

TEST(NestedKFoldTest, InvalidParametersThrow)
{
    EXPECT_THROW(
        {
            const auto out = statistics::NestedKFold(1U, 2U, false, 1U).split(10U);
            (void) out;
        },
        std::invalid_argument);

    EXPECT_THROW(
        {
            // inner_n for each outer fold will be 8, so 9 is invalid.
            const auto out = statistics::NestedKFold(5U, 9U, false, 1U).split(10U);
            (void) out;
        },
        std::invalid_argument);
}
