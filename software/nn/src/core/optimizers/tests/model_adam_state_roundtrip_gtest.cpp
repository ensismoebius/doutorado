/**
 * @file src/core/optimizers/tests/model_adam_state_roundtrip_gtest.cpp
 * @brief Implementation for Model adam state roundtrip gtest.
 *

 */

#include <gtest/gtest.h>

#include "nn/io/StateIO.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/testing/tempfile.hpp"

using namespace nn;

TEST(AdamModelState, SaveLoadModelAndOptimizerRoundtrip)
{
    // Build a tiny model: Sequential { Linear(3->2), Linear(2->1) }
    auto l1 = std::make_shared<Linear>(3, 2);
    auto l2 = std::make_shared<Linear>(2, 1);
    Sequential model({l1, l2});

    // Initialize a dummy input and perform a forward/backward to populate grads
    Tensor inp(1, 3);
    inp.at(0, 0) = 1.0f;
    inp.at(0, 1) = 2.0f;
    inp.at(0, 2) = 3.0f;

    auto out = model.forward(inp, true);
    // Create a simple loss: sum of outputs
    Tensor loss_grad(out.rows(), out.cols());
    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j) loss_grad.at(i, j) = 1.0f;
    model.backward(loss_grad);

    // Attach Adam to model params and do a step to create optimizer state
    Adam adam(0.01f);
    auto pspan = model.params();
    std::vector<Tensor*> params_vec(pspan.begin(), pspan.end());
    adam.attach(params_vec);
    adam.step(params_vec);

    // Save combined state: model.state_dict + optimizer.state_dict prefixed
    auto m_sd = model.state_dict();
    auto o_sd = adam.state_dict();
    // Merge with prefixes
    std::map<std::string, Tensor> combined;
    for (const auto& kv : m_sd) combined[std::string("model.") + kv.first] = kv.second;
    for (const auto& kv : o_sd) combined[std::string("optim.") + kv.first] = kv.second;

    const std::string tmp = nn::testing::make_temp_file("model_adam_");
    ASSERT_FALSE(tmp.empty());
    ASSERT_TRUE(nn::io::save_state_dict(combined, tmp));

    auto loaded = nn::io::load_state_dict(tmp);
    ASSERT_EQ(loaded.size(), combined.size());
    // Check a few keys exist
    ASSERT_TRUE(loaded.count("model.0.weight") || loaded.count("model.0.bias") ||
                loaded.count("model.1.weight"));
    ASSERT_TRUE(loaded.count("optim.time_step"));

    std::remove(tmp.c_str());
}
