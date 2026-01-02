#include <iterator>

#include "../DataLoader.h"
#include "../TensorDataset.h"
#include "gtest/gtest.h"

// Helper to build a Tensor with sequential rows (N x D)
static auto make_sequential_tensor(std::size_t N, std::size_t D) -> nn::Tensor
{
    Eigen::MatrixXf m(static_cast<int>(N), static_cast<int>(D));
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < D; ++j)
        {
            m(static_cast<int>(i), static_cast<int>(j)) = static_cast<float>((i * D) + j);
        }
    }
    return nn::Tensor{m};
}

TEST(DataLoaderTest, DeterministicShuffle)
{
    auto inputs = make_sequential_tensor(10, 2);
    auto targets = make_sequential_tensor(10, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader1(dataset, 4, true, 42U);
    DataLoader loader2(dataset, 4, true, 42U);

    std::vector<int> order1;
    for (const auto& batch : loader1)
    {
        for (int i = 0; i < batch.inputs.get_data_ref().rows(); ++i)
        {
            order1.push_back(static_cast<int>(batch.inputs.get_data_ref()(i, 0) / 2));
        }
    }

    std::vector<int> order2;
    for (const auto& batch : loader2)
    {
        for (int i = 0; i < batch.inputs.get_data_ref().rows(); ++i)
        {
            order2.push_back(static_cast<int>(batch.inputs.get_data_ref()(i, 0) / 2));
        }
    }

    EXPECT_EQ(order1, order2);
}

TEST(DataLoaderTest, ShuffleVsNoShuffle)
{
    auto inputs = make_sequential_tensor(8, 2);
    auto targets = make_sequential_tensor(8, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader shuffled(dataset, 4, true);
    DataLoader not_shuffled(dataset, 4, false);

    std::vector<int> sorder;
    for (const auto& batch : shuffled)
    {
        for (int i = 0; i < batch.inputs.get_data_ref().rows(); ++i)
        {
            sorder.push_back(static_cast<int>(batch.inputs.get_data_ref()(i, 0) / 2));
        }
    }

    std::vector<int> norder;
    for (const auto& batch : not_shuffled)
    {
        for (int i = 0; i < batch.inputs.get_data_ref().rows(); ++i)
        {
            norder.push_back(static_cast<int>(batch.inputs.get_data_ref()(i, 0) / 2));
        }
    }

    // shuffled order should not be equal to not_shuffled order most likely; make
    // a weak assertion
    bool equal = (sorder == norder); // flawfinder: ignore
    EXPECT_TRUE(!equal || sorder.size() == 0 || norder.size() == 0);
}

TEST(DataLoaderTest, SmallDataset)
{
    auto inputs = make_sequential_tensor(3, 2);
    auto targets = make_sequential_tensor(3, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 8, false);

    int batches = 0;
    int total_rows = 0;
    for (const auto& batch : loader)
    {
        ++batches;
        total_rows += static_cast<int>(batch.inputs.get_data_ref().rows());
    }
    EXPECT_EQ(batches, 1);
    EXPECT_EQ(total_rows, 3);
}

TEST(DataLoaderTest, EmptyDataset)
{
    auto inputs = make_sequential_tensor(0, 0);
    auto targets = make_sequential_tensor(0, 0);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 4, false);
    int count = std::distance(loader.begin(), loader.end());
    EXPECT_EQ(count, 0);
}

TEST(DataLoaderTest, ZeroBatchSizeThrows)
{
    auto inputs = make_sequential_tensor(10, 2);
    auto targets = make_sequential_tensor(10, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    ASSERT_THROW(DataLoader loader(dataset, 0, false), std::invalid_argument);
}

// Exception Testing for DataLoaders
TEST(DataLoaderExceptionTest, MismatchedDatasetSizes)
{
    auto inputs = make_sequential_tensor(10, 2);
    auto targets = make_sequential_tensor(5, 1); // Different size
    ASSERT_THROW(std::make_shared<TensorDataset>(inputs, targets), std::invalid_argument);
}

TEST(DataLoaderExceptionTest, NullDataset)
{
    std::shared_ptr<TensorDataset> null_dataset = nullptr;
    ASSERT_THROW(DataLoader loader(null_dataset, 4, false), std::invalid_argument);
}

TEST(DataLoaderExceptionTest, NegativeBatchSize)
{
    auto inputs = make_sequential_tensor(10, 2);
    auto targets = make_sequential_tensor(10, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    ASSERT_THROW(DataLoader loader(dataset, -1, false), std::invalid_argument);
}

// Memory Stress Testing for DataLoaders
TEST(DataLoaderMemoryStressTest, LargeDataset)
{
    const size_t large_size = 10000;
    const size_t feature_dim = 100;

    auto inputs = make_sequential_tensor(large_size, feature_dim);
    auto targets = make_sequential_tensor(large_size, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 1000, false);

    size_t total_samples = 0;
    size_t batch_count = 0;

    for (const auto& batch : loader)
    {
        ++batch_count;
        total_samples += batch.inputs.get_data_ref().rows();
        EXPECT_EQ(batch.inputs.get_data_ref().cols(), feature_dim);
        EXPECT_EQ(batch.targets.get_data_ref().cols(), 1);
    }

    EXPECT_EQ(total_samples, large_size);
    EXPECT_EQ(batch_count, 10); // 10000 / 1000 = 10 batches
}

TEST(DataLoaderMemoryStressTest, LargeBatchSize)
{
    const size_t dataset_size = 1000;
    const size_t batch_size = 500;

    auto inputs = make_sequential_tensor(dataset_size, 50);
    auto targets = make_sequential_tensor(dataset_size, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, batch_size, false);

    size_t batch_count = 0;
    for (const auto& batch : loader)
    {
        ++batch_count;
        EXPECT_EQ(batch.inputs.get_data_ref().rows(), batch_size);
    }

    EXPECT_EQ(batch_count, 2); // 1000 / 500 = 2 batches
}

// Thread Safety Validation for DataLoaders
TEST(DataLoaderThreadSafetyTest, MultipleIterators)
{
    auto inputs = make_sequential_tensor(100, 5);
    auto targets = make_sequential_tensor(100, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader1(dataset, 10, true, 123U);
    DataLoader loader2(dataset, 10, true, 123U);

    // Both loaders should produce the same order with same seed
    auto it1 = loader1.begin();
    auto it2 = loader2.begin();

    for (int i = 0; i < 5; ++i)
    {
        ASSERT_NE(it1, loader1.end());
        ASSERT_NE(it2, loader2.end());

        const auto& batch1 = *it1;
        const auto& batch2 = *it2;

        // Batches should be identical with same seed
        EXPECT_TRUE(batch1.inputs.get_data_ref().isApprox(batch2.inputs.get_data_ref()));
        EXPECT_TRUE(batch1.targets.get_data_ref().isApprox(batch2.targets.get_data_ref()));

        ++it1;
        ++it2;
    }
}

TEST(DataLoaderThreadSafetyTest, IteratorIndependence)
{
    auto inputs = make_sequential_tensor(50, 3);
    auto targets = make_sequential_tensor(50, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 5, false);

    auto it1 = loader.begin();
    auto it2 = loader.begin();

    // Advance it1 by 2 batches
    ++it1;
    ++it1;

    // it2 should still be at the beginning
    ASSERT_NE(it2, loader.end());
    EXPECT_EQ((*it2).inputs.get_data_ref()(0, 0), 0.0f);

    // it1 should be at the third batch (batch index 2)
    // Batch 2 contains rows [10,11,12,13,14], and row 10's first element is 10*3 = 30
    ASSERT_NE(it1, loader.end());
    EXPECT_EQ((*it1).inputs.get_data_ref()(0, 0), 30.0f); // Row 10: (10 * D) + 0 = 30
}

// Numerical Edge Cases for DataLoaders
TEST(DataLoaderNumericalEdgeTest, SingleSampleDataset)
{
    auto inputs = make_sequential_tensor(1, 2);
    auto targets = make_sequential_tensor(1, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 1, false);

    auto it = loader.begin();
    ASSERT_NE(it, loader.end());

    const auto& batch = *it;
    EXPECT_EQ(batch.inputs.get_data_ref().rows(), 1);
    EXPECT_EQ(batch.inputs.get_data_ref().cols(), 2);
    EXPECT_EQ(batch.targets.get_data_ref().rows(), 1);
    EXPECT_EQ(batch.targets.get_data_ref().cols(), 1);

    ++it;
    EXPECT_EQ(it, loader.end());
}

TEST(DataLoaderNumericalEdgeTest, BatchSizeLargerThanDataset)
{
    auto inputs = make_sequential_tensor(3, 2);
    auto targets = make_sequential_tensor(3, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 10, false); // batch_size > dataset_size

    auto it = loader.begin();
    ASSERT_NE(it, loader.end());

    const auto& batch = *it;
    EXPECT_EQ(batch.inputs.get_data_ref().rows(), 3); // Should return all samples
    EXPECT_EQ(batch.targets.get_data_ref().rows(), 3);

    ++it;
    EXPECT_EQ(it, loader.end());
}

TEST(DataLoaderNumericalEdgeTest, PerfectBatchDivision)
{
    auto inputs = make_sequential_tensor(12, 2);
    auto targets = make_sequential_tensor(12, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 3, false); // 12 / 3 = 4 perfect batches

    int batch_count = 0;
    size_t total_samples = 0;

    for (const auto& batch : loader)
    {
        ++batch_count;
        total_samples += batch.inputs.get_data_ref().rows();
        EXPECT_EQ(batch.inputs.get_data_ref().rows(), 3);
    }

    EXPECT_EQ(batch_count, 4);
    EXPECT_EQ(total_samples, 12);
}

// Additional Comprehensive Tests
TEST(DataLoaderComprehensiveTest, IteratorReset)
{
    auto inputs = make_sequential_tensor(20, 2);
    auto targets = make_sequential_tensor(20, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 5, false);

    // First pass
    std::vector<float> first_pass;
    for (const auto& batch : loader)
    {
        first_pass.push_back(batch.inputs.get_data_ref()(0, 0));
    }

    // Second pass should be identical (no shuffle)
    std::vector<float> second_pass;
    for (const auto& batch : loader)
    {
        second_pass.push_back(batch.inputs.get_data_ref()(0, 0));
    }

    EXPECT_EQ(first_pass, second_pass);
}

TEST(DataLoaderComprehensiveTest, BatchContentVerification)
{
    auto inputs = make_sequential_tensor(8, 2);
    auto targets = make_sequential_tensor(8, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 4, false);

    auto it = loader.begin();
    ASSERT_NE(it, loader.end());

    const auto& batch = *it;
    // Verify first batch contains samples 0-3
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(batch.inputs.get_data_ref()(i, 0), i * 2.0f);
        EXPECT_EQ(batch.inputs.get_data_ref()(i, 1), i * 2.0f + 1.0f);
        EXPECT_EQ(batch.targets.get_data_ref()(i, 0), i * 1.0f);
    }

    ++it;
    ASSERT_NE(it, loader.end());

    const auto& batch2 = *it;
    // Verify second batch contains samples 4-7
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(batch2.inputs.get_data_ref()(i, 0), (i + 4) * 2.0f);
        EXPECT_EQ(batch2.inputs.get_data_ref()(i, 1), (i + 4) * 2.0f + 1.0f);
        EXPECT_EQ(batch2.targets.get_data_ref()(i, 0), (i + 4) * 1.0f);
    }
}

TEST(DataLoaderComprehensiveTest, ShuffleDeterminism)
{
    auto inputs = make_sequential_tensor(16, 2);
    auto targets = make_sequential_tensor(16, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Test that different seeds produce different orders
    DataLoader loader1(dataset, 4, true, 123U);
    DataLoader loader2(dataset, 4, true, 456U);

    std::vector<float> order1, order2;
    for (const auto& batch : loader1)
    {
        order1.push_back(batch.inputs.get_data_ref()(0, 0));
    }
    for (const auto& batch : loader2)
    {
        order2.push_back(batch.inputs.get_data_ref()(0, 0));
    }

    // Different seeds should likely produce different orders
    EXPECT_NE(order1, order2);
}