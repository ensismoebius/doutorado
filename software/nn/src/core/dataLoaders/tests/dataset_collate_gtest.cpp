/**
 * @file src/core/dataLoaders/tests/dataset_collate_gtest.cpp
 * @brief Implementation for Dataset collate gtest.
 *

 */

#include <gtest/gtest.h>

#include "nn/dataLoaders/Dataset.hpp"

// Minimal test dataset that returns single-row input/target tensors per index
class TestDataset : public Dataset
{
   public:
    explicit TestDataset(std::size_t n, std::size_t in_features, std::size_t tg_features)
        : n_(n), in_(in_features), tg_(tg_features)
    {
    }

    auto get_item(std::size_t idx) const -> Batch override
    {
        // Return inputs = [idx, idx, ...] single-row tensor, targets = [idx]
        nn::Tensor in(1, static_cast<nn::Index>(in_));
        nn::Tensor tg(1, static_cast<nn::Index>(tg_));
        for (std::size_t c = 0; c < in_; ++c)
            in.at(0, static_cast<nn::Index>(c)) = static_cast<float>(idx);
        for (std::size_t c = 0; c < tg_; ++c)
            tg.at(0, static_cast<nn::Index>(c)) = static_cast<float>(idx);
        return Batch{.inputs = std::move(in), .targets = std::move(tg)};
    }

    auto size() const -> std::size_t override
    {
        return n_;
    }

   private:
    std::size_t n_;
    std::size_t in_;
    std::size_t tg_;
};

TEST(DatasetCollate, CollatesMultipleIndicesInOrder)
{
    auto ds = std::make_shared<TestDataset>(5, 3, 2);

    // Indices in non-trivial order
    std::vector<std::size_t> indices = {2, 0, 4};

    Batch b = ds->collate(indices);

    EXPECT_EQ(b.inputs.rows(), static_cast<nn::Index>(indices.size()));
    EXPECT_EQ(b.targets.rows(), static_cast<nn::Index>(indices.size()));
    EXPECT_EQ(b.inputs.cols(), static_cast<nn::Index>(3));
    EXPECT_EQ(b.targets.cols(), static_cast<nn::Index>(2));

    // Check that each row contains the index value used in get_item
    for (std::size_t r = 0; r < indices.size(); ++r)
    {
        float expected = static_cast<float>(indices[r]);
        for (nn::Index c = 0; c < b.inputs.cols(); ++c)
        {
            EXPECT_FLOAT_EQ(b.inputs.at(static_cast<nn::Index>(r), c), expected);
        }
        for (nn::Index c = 0; c < b.targets.cols(); ++c)
        {
            EXPECT_FLOAT_EQ(b.targets.at(static_cast<nn::Index>(r), c), expected);
        }
    }
}

TEST(DatasetCollate, SizeReportsConfiguredLength)
{
    TestDataset ds(7, 3, 2);
    EXPECT_EQ(ds.size(), 7U);
}
