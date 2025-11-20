#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "core/tensor/Tensor.hpp"
#include "../batching.hpp"
#include "../synthetic_spike_data.hpp"
#include "../vectorizationCheck.hpp"

// Util: synthetic_spike_data
TEST(UtilTest, SyntheticSpikeData)
{
    int n_samples = 5;
    int input_dim = 3;
    int n_steps = 4;
    float max_rate = 1.0F;
    float timeStep = 1.0F;

    auto [spike_trains, real_valued] =
        generate_autoencoder_spike_data(n_samples, input_dim, n_steps, max_rate, timeStep);

    ASSERT_EQ(spike_trains.size(), n_steps);
    for (const auto& spikes : spike_trains)
    {
        ASSERT_EQ(spikes.get_data_ref().rows(), n_samples);
        ASSERT_EQ(spikes.get_data_ref().cols(), input_dim);
        for (int i = 0; i < spikes.get_data_ref().rows(); ++i)
        {
            for (int j = 0; j < spikes.get_data_ref().cols(); ++j)
            {
                ASSERT_TRUE(spikes.get_data_ref()(i, j) == 0.0F || spikes.get_data_ref()(i, j) == 1.0F);
            }
        }
    }
    ASSERT_EQ(real_valued.size(), n_steps);
}

// Util: vectorizationCheck
TEST(UtilTest, VectorizationCheck)
{
    ASSERT_NO_THROW(printVectorizationSupport());
}

// Util: batching
TEST(UtilTest, Batching)
{
    std::vector<Tensor> input_samples;
    std::vector<Tensor> target_samples;
    for (int i = 0; i < 4; ++i)
    {
        input_samples.push_back(Tensor(Eigen::MatrixXf::Random(1, 2)));
        target_samples.push_back(Tensor(Eigen::MatrixXf::Random(1, 1)));
    }

    auto batches = create_batches(input_samples, target_samples, 2);

    ASSERT_EQ(batches.size(), 2U);
    ASSERT_EQ(batches[0].inputs.get_data_ref().rows(), 2);
}
