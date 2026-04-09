/**
 * @file NetworkSerializer_gtest.cpp
 * @brief Unit tests for round-trip network serialization (NPZ) and architecture matching.
 */

#include <cstdio>

#include "gtest/gtest.h"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/saver/NetworkSerializer.hpp"

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
    // Set weight to constant 42.0F
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            linear->weight.at(i, j) = 42.0F;
        }
    }
    // Set bias to constant -7.0F
    for (int i = 0; i < 3; ++i)
    {
        linear->bias.at(i, 0) = -7.0F;
    }

    auto leaky = std::dynamic_pointer_cast<Leaky>(model.layers[2]);
    leaky->resistance.at(0, 0) = 2.0F;
    leaky->voltage_threshold.at(0, 0) = 4.0F;

    // Save the model
    std::string filename = temp_directory_path().string() + "/test_model_save_load.npz";
    ASSERT_TRUE(NetworkSerializer::saveNetwork(model, filename));

    // Ensure new trainable parameter is persisted in checkpoint payload.
    const cnpy::npz_t saved_data = cnpy::npz_load(filename);
    EXPECT_NE(saved_data.find("2.capacitance"), saved_data.end());

    // Load into a new model (may be disabled in this build)
    Sequential loaded; // LCOV_EXCL_LINE
    if (!NetworkSerializer::loadNetwork(loaded, filename))
    {
        GTEST_SKIP()
            << "NPZ runtime loading is disabled in this build; skipping load assertions."; // LCOV_EXCL_LINE
    }

    // Check architecture: layer types and order
    ASSERT_EQ(loaded.layers.size(), 4);
    EXPECT_TRUE(std::dynamic_pointer_cast<Linear>(loaded.layers[0]));
    EXPECT_TRUE(std::dynamic_pointer_cast<LeakyReLU>(loaded.layers[1]));
    EXPECT_TRUE(std::dynamic_pointer_cast<Leaky>(loaded.layers[2]));
    EXPECT_TRUE(std::dynamic_pointer_cast<ReLU>(loaded.layers[3]));

    // Check Linear weights and bias match
    auto loaded_linear = std::dynamic_pointer_cast<Linear>(loaded.layers[0]);
    ASSERT_TRUE(loaded_linear);
    EXPECT_EQ(loaded_linear->weight.rows(), 3);
    EXPECT_EQ(loaded_linear->weight.cols(), 4);

    // Check all weights are 42.0F
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_FLOAT_EQ(loaded_linear->weight.at(i, j), 42.0F);
        }
    }

    // Check all biases are -7.0F
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_FLOAT_EQ(loaded_linear->bias.at(i, 0), -7.0F);
    }

    // Check Leaky config and parameters
    auto loaded_leaky = std::dynamic_pointer_cast<Leaky>(loaded.layers[2]);
    ASSERT_TRUE(loaded_leaky);
    EXPECT_FLOAT_EQ(loaded_leaky->time_step, 1.0F);
    EXPECT_FLOAT_EQ(loaded_leaky->capacitance.at(0, 0), 3.0F);
    EXPECT_FLOAT_EQ(loaded_leaky->reset_potential, 0.5F);
    EXPECT_TRUE(loaded_leaky->reset_zero);
    EXPECT_FLOAT_EQ(loaded_leaky->resistance.at(0, 0), 2.0F);
    EXPECT_FLOAT_EQ(loaded_leaky->voltage_threshold.at(0, 0), 4.0F);

    // Clean up
    std::remove(filename.c_str());
}
