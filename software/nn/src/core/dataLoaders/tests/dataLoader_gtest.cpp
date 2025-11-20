#include "../DataLoader.h"
#include "../TensorDataset.h"
#include "gtest/gtest.h"

// Helper to build a Tensor with sequential rows (N x D)
static auto make_sequential_tensor(std::size_t N, std::size_t D) -> Tensor
{
    Eigen::MatrixXf m(static_cast<int>(N), static_cast<int>(D));
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < D; ++j)
        {
            m(static_cast<int>(i), static_cast<int>(j)) = static_cast<float>((i * D) + j);
        }
    }
    return Tensor{m};
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
    bool equal = (sorder == norder);
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
    int count = 0;
    for (const auto& batch : loader)
    {
        ++count;
    }
    EXPECT_EQ(count, 0);
}
