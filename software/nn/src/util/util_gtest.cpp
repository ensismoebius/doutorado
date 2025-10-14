#include "batching.hpp"
#include "synthetic_spike_data.hpp"
#include "../tensor/Tensor.hpp"
#include "vectorizationCheck.hpp"
#include <Eigen/Dense>
#include <gtest/gtest.h>

// Util: synthetic_spike_data
TEST(UtilTest, SyntheticSpikeData)
{
  int n_samples = 5;
  int input_dim = 3;
  int n_steps = 4;
  float max_rate = 1.0F;
  float timeStep = 1.0F;

  auto [spike_trains, real_valued] = generate_autoencoder_spike_data(n_samples, input_dim, n_steps, max_rate, timeStep);

  ASSERT_EQ(spike_trains.size(), n_steps);
  for (const auto &spikes : spike_trains)
  {
    ASSERT_EQ(spikes.data.rows(), n_samples);
    ASSERT_EQ(spikes.data.cols(), input_dim);
    for (int i = 0; i < spikes.data.rows(); ++i)
    {
      for (int j = 0; j < spikes.data.cols(); ++j)
      {
        ASSERT_TRUE(spikes.data(i, j) == 0.0F || spikes.data(i, j) == 1.0F);
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
  for (int i = 0; i < 4; ++i) {
    input_samples.push_back(Tensor(Eigen::MatrixXf::Random(1, 2)));
    target_samples.push_back(Tensor(Eigen::MatrixXf::Random(1, 1)));
  }

  auto batches = create_batches(input_samples, target_samples, 2);

  ASSERT_EQ(batches.size(), 2U);
  ASSERT_EQ(batches[0].inputs.data.rows(), 2);
}
