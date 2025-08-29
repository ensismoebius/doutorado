// #include "batching.hpp"
// #include "synthetic_spike_data.hpp"
// #include "tensor/Tensor.hpp"
// #include "vectorizationCheck.hpp"
// #include <Eigen/Dense>
// #include <gtest/gtest.h>
// // Util: synthetic_spike_data
// TEST(UtilTest, SyntheticSpikeData)
// {

//   int n_samples = 5;
//   int input_dim = 3;
//   int n_steps = 4;

//   float const max_rate = 1.0F;
//   float const timeStep = 1.0F;

//   Eigen::MatrixXf real_valued;

//   auto spike_trains = generate_synthetic_spike_data(n_samples, input_dim, n_steps, max_rate,
//   timeStep);

//   ASSERT_EQ(spike_trains.size(), n_steps);
//   for (const auto &spikes : spike_trains)
//     {
//       ASSERT_EQ(spikes.rows(), n_samples);
//       ASSERT_EQ(spikes.cols(), input_dim);
//       for (int i = 0; i < spikes.rows(); ++i)
//         {
//           for (int j = 0; j < spikes.cols(); ++j)
//             {
//               ASSERT_TRUE(spikes(i, j) == 0.0F || spikes(i, j) == 1.0F);
//             }
//         }
//     }
//   ASSERT_EQ(real_valued.rows(), n_samples);
//   ASSERT_EQ(real_valued.cols(), input_dim);
// }

// // Util: vectorizationCheck
// TEST(UtilTest, VectorizationCheck)
// {
//   ASSERT_NO_THROW(printVectorizationSupport());
// }

// // Util: batching
// TEST(UtilTest, Batching)
// {
//   Eigen::MatrixXf input_matrix = Eigen::MatrixXf::Random(4, 2);
//   Eigen::MatrixXf target_matrix = Eigen::MatrixXf::Random(4, 1);
//   Tensor input(input_matrix);
//   Tensor target(target_matrix);

//   auto batches = create_batches(input, target, 2);

//   ASSERT_EQ(batches.size(), 2U);
//   ASSERT_EQ(batches[0].inputs.data.rows(), 2);
// }
