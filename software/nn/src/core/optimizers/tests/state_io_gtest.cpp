/**
 * @file src/core/optimizers/tests/state_io_gtest.cpp
 * @brief Implementation for State io gtest.
 *

 */

#include <gtest/gtest.h>

#include <cstdio>

#include "io/StateIO.hpp"
#include "optimizers/Adam.hpp"
#include "tensor/Tensor.hpp"
#include "test_utils/tempfile.hpp"

using namespace nn;

TEST(StateIOTest, SaveLoadSimpleStateDict)
{
    std::map<std::string, Tensor> sd;
    Tensor w(2, 3);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) w.at(i, j) = static_cast<float>(i * 10 + j);
    Tensor b(2, 1);
    b.at(0, 0) = 1.5f;
    b.at(1, 0) = -2.25f;
    sd["layer.weight"] = w;
    sd["layer.bias"] = b;

    const std::string tmp = nn::testing::make_temp_file("stateio_");
    ASSERT_FALSE(tmp.empty());
    ASSERT_TRUE(nn::io::save_state_dict(sd, tmp));

    std::map<std::string, Tensor> loaded;
    ASSERT_TRUE(nn::io::load_state_dict(loaded, tmp));

    ASSERT_EQ(loaded.size(), sd.size());
    for (const auto& kv : sd)
    {
        auto it = loaded.find(kv.first);
        ASSERT_NE(it, loaded.end());
        const Tensor& a = kv.second;
        const Tensor& loaded_tensor = it->second;
        ASSERT_EQ(a.rows(), loaded_tensor.rows());
        ASSERT_EQ(a.cols(), loaded_tensor.cols());
        for (size_t i = 0; i < a.rows(); ++i)
            for (size_t j = 0; j < a.cols(); ++j)
                ASSERT_FLOAT_EQ(a.at(i, j), loaded_tensor.at(i, j));
    }

    std::remove(tmp.c_str());
}

TEST(StateIOTest, AdamStateDictRoundtrip)
{
    // Create a parameter and an Adam instance, perform a step to populate moments
    Tensor p(3, 3);
    p.set_grad(Tensor::constant(3, 3, 0.1f));
    std::vector<Tensor*> params = {&p};

    Adam adam(0.01f);
    adam.attach(params);
    adam.step(params);
    auto sd = adam.state_dict();

    const std::string tmp = nn::testing::make_temp_file("adamstate_");
    ASSERT_FALSE(tmp.empty());
    ASSERT_TRUE(nn::io::save_state_dict(sd, tmp));

    std::map<std::string, Tensor> loaded;
    ASSERT_TRUE(nn::io::load_state_dict(loaded, tmp));

    // Ensure keys match and time_step present
    ASSERT_NE(loaded.find("time_step"), loaded.end());
    for (const auto& kv : sd)
    {
        auto it = loaded.find(kv.first);
        ASSERT_NE(it, loaded.end());
        const Tensor& a = kv.second;
        const Tensor& loaded_tensor = it->second;
        ASSERT_EQ(a.rows(), loaded_tensor.rows());
        ASSERT_EQ(a.cols(), loaded_tensor.cols());
    }

    std::remove(tmp.c_str());
}
