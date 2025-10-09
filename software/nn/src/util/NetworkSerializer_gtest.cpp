#include "layers/Leaky.hpp"
#include "layers/LeakyReLU.hpp"
#include "layers/Linear.hpp"
#include "layers/ReLU.hpp"
#include "layers/Sequential.hpp"
#include "util/NetworkSerializer.hpp"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <cstdio>

TEST(NetworkSerializerTest, SaveLoadRoundTripMatchesPyTorchStandard) {
  // Build a model with all supported layers
  Sequential model;
  model.layers.push_back(std::make_shared<Linear>(4, 3));
  model.layers.push_back(std::make_shared<LeakyReLU>(0.2f));
  model.layers.push_back(std::make_shared<Leaky>(1.0f, 2.0f, 3.0f, 4.0f, true, 0.5f));
  model.layers.push_back(std::make_shared<ReLU>());

  // Fill weights and biases with known values for deterministic test
  auto linear = std::dynamic_pointer_cast<Linear>(model.layers[0]);
  linear->weight.data.setConstant(42.0f);
  linear->bias.data.setConstant(-7.0f);

  auto leaky = std::dynamic_pointer_cast<Leaky>(model.layers[2]);
  leaky->resistance.data.setConstant(2.0f);
  leaky->voltage_threshold.data.setConstant(4.0f);

  // Save the model
  std::string filename = "test_model_save_load.npz";
  ASSERT_TRUE(NetworkSerializer::saveNetwork(model, filename));

  // Load into a new model
  Sequential loaded;
  ASSERT_TRUE(NetworkSerializer::loadNetwork(loaded, filename));

  // Check architecture: layer types and order
  ASSERT_EQ(loaded.layers.size(), 4);
  EXPECT_TRUE(std::dynamic_pointer_cast<Linear>(loaded.layers[0]));
  EXPECT_TRUE(std::dynamic_pointer_cast<LeakyReLU>(loaded.layers[1]));
  EXPECT_TRUE(std::dynamic_pointer_cast<Leaky>(loaded.layers[2]));
  EXPECT_TRUE(std::dynamic_pointer_cast<ReLU>(loaded.layers[3]));

  // Check Linear weights and bias match
  auto loaded_linear = std::dynamic_pointer_cast<Linear>(loaded.layers[0]);
  ASSERT_TRUE(loaded_linear);
  EXPECT_EQ(loaded_linear->weight.data.rows(), 3);
  EXPECT_EQ(loaded_linear->weight.data.cols(), 4);
  EXPECT_TRUE(loaded_linear->weight.data.isApprox(Eigen::MatrixXf::Constant(3, 4, 42.0f)));
  EXPECT_TRUE(loaded_linear->bias.data.isApprox(Eigen::MatrixXf::Constant(3, 1, -7.0f)));

  // Check Leaky config and parameters
  auto loaded_leaky = std::dynamic_pointer_cast<Leaky>(loaded.layers[2]);
  ASSERT_TRUE(loaded_leaky);
  EXPECT_FLOAT_EQ(loaded_leaky->dt, 1.0f);
  EXPECT_FLOAT_EQ(loaded_leaky->capacitance, 3.0f);
  EXPECT_FLOAT_EQ(loaded_leaky->reset_potential, 0.5f);
  EXPECT_TRUE(loaded_leaky->reset_zero);
  EXPECT_TRUE(loaded_leaky->resistance.data.isApprox(Eigen::MatrixXf::Constant(1, 1, 2.0f)));
  EXPECT_TRUE(loaded_leaky->voltage_threshold.data.isApprox(Eigen::MatrixXf::Constant(1, 1, 4.0f)));

  // Clean up
  std::remove(filename.c_str());
}
