#include <Eigen/Dense>
#include <cstdio>

#include "gtest/gtest.h"
#include "core/layers/Leaky.hpp"
#include "core/layers/LeakyReLU.hpp"
#include "core/layers/Linear.hpp"
#include "core/layers/ReLU.hpp"
#include "core/layers/Sequential.hpp"
#include "util/NetworkSerializer.hpp"

using std::filesystem::temp_directory_path;

TEST(NetworkSerializerTest, SaveLoadRoundTripMatchesPyTorchStandard)
{
    // Build a model with all supported layers
    Sequential model;
    model.layers.push_back(std::make_shared<Linear>(4, 3));
    model.layers.push_back(std::make_shared<LeakyReLU>(0.2F));
    model.layers.push_back(std::make_shared<Leaky>(1.0F, 2.0F, 3.0F, 4.0F, true, 0.5F));
    model.layers.push_back(std::make_shared<ReLU>());

    // Fill weights and biases with known values for deterministic test
    auto linear = std::dynamic_pointer_cast<Linear>(model.layers[0]);
    linear->weight.data.setConstant(42.0F);
    linear->bias.data.setConstant(-7.0F);

    auto leaky = std::dynamic_pointer_cast<Leaky>(model.layers[2]);
    leaky->resistance.data.setConstant(2.0F);
    leaky->voltage_threshold.data.setConstant(4.0F);

    // Save the model
    std::string filename = temp_directory_path().string() + "/test_model_save_load.npz";
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

    Eigen::MatrixXf weight_data = Eigen::MatrixXf::Constant(3, 4, 42.0F);
    EXPECT_TRUE(loaded_linear->weight.data.isApprox(weight_data));

    Eigen::MatrixXf bias_data = Eigen::MatrixXf::Constant(3, 1, -7.0F);
    EXPECT_TRUE(loaded_linear->bias.data.isApprox(bias_data));

    // Check Leaky config and parameters
    auto loaded_leaky = std::dynamic_pointer_cast<Leaky>(loaded.layers[2]);
    ASSERT_TRUE(loaded_leaky);
    EXPECT_FLOAT_EQ(loaded_leaky->dt, 1.0F);
    EXPECT_FLOAT_EQ(loaded_leaky->capacitance, 3.0F);
    EXPECT_FLOAT_EQ(loaded_leaky->reset_potential, 0.5F);
    EXPECT_TRUE(loaded_leaky->reset_zero);
    EXPECT_TRUE(loaded_leaky->resistance.data.isApprox(Eigen::MatrixXf::Constant(1, 1, 2.0F)));
    EXPECT_TRUE(
        loaded_leaky->voltage_threshold.data.isApprox(Eigen::MatrixXf::Constant(1, 1, 4.0F)));

    // Clean up
    std::remove(filename.c_str());
}
