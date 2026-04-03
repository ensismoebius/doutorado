/**
 * @file src/core/serialization/tests/StateIO_gtest.cpp
 * @brief Implementation for Stateio gtest.
 *

 */

#include <gtest/gtest.h>

#include "nn/serialization/StateIO.hpp"
#include "nn/testing/tempfile.hpp"

using namespace nn::serialization;

TEST(StateIO, SaveLoadRoundtrip)
{
    StateDict sd;
    // Create two small tensors
    nn::Tensor a(2, 2);
    float valsA[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 4; ++i) a.mutable_data_ptr()[i] = valsA[i];

    nn::Tensor b(1, 3);
    float valsB[3] = {0.5f, -1.25f, 2.5f};
    for (int i = 0; i < 3; ++i) b.mutable_data_ptr()[i] = valsB[i];

    sd.emplace("weights", a);
    sd.emplace("bias", b);

    // Temp file
    nn::testing::TempFile tf("nn_state_");
    ASSERT_FALSE(tf.path().empty());

    bool ok = save_state_dict(sd, tf.path());
    ASSERT_TRUE(ok);

    StateDict loaded = load_state_dict(tf.path());
    ASSERT_EQ(loaded.size(), sd.size());

    auto it = loaded.find("weights");
    ASSERT_NE(it, loaded.end());
    const nn::Tensor& la = it->second;
    ASSERT_EQ(la.rows(), 2);
    ASSERT_EQ(la.cols(), 2);
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(la.data_ptr()[i], valsA[i]);

    it = loaded.find("bias");
    ASSERT_NE(it, loaded.end());
    const nn::Tensor& lb = it->second;
    ASSERT_EQ(lb.rows(), 1);
    ASSERT_EQ(lb.cols(), 3);
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(lb.data_ptr()[i], valsB[i]);
}
