/**
 * @file src/core/saver/tests/StateIO_gtest.cpp
 * @brief Implementation for Stateio gtest.
 *

 */

#include <gtest/gtest.h>

#include <filesystem>

#include "nn/serialization/StateIO.hpp"

using namespace nn::serialization;

TEST(StateIO, RoundtripSaveLoad)
{
    StateDict sd;
    nn::Tensor a(2, 3);
    // fill a with 0..5
    for (nn::Index i = 0; i < a.rows(); ++i)
        for (nn::Index j = 0; j < a.cols(); ++j) a.at(i, j) = static_cast<float>(i * a.cols() + j);
    nn::Tensor b(1, 4);
    for (nn::Index j = 0; j < b.cols(); ++j) b.at(0, j) = 0.5f + static_cast<float>(j);
    sd["layer1.weight"] = a;
    sd["layer1.bias"] = b;

    const std::string tmpdir = (std::filesystem::temp_directory_path() / "stateio_test").string();
    std::filesystem::create_directories(tmpdir);
    const std::string path = tmpdir + "/state.bin";

    ASSERT_TRUE(save_state_dict(sd, path));

    StateDict loaded = load_state_dict(path);
    ASSERT_EQ(loaded.size(), sd.size());
    for (const auto& kv : sd)
    {
        auto it = loaded.find(kv.first);
        ASSERT_NE(it, loaded.end());
        const nn::Tensor& orig = kv.second;
        const nn::Tensor& got = it->second;
        EXPECT_EQ(orig.rows(), got.rows());
        EXPECT_EQ(orig.cols(), got.cols());
        for (nn::Index i = 0; i < orig.rows(); ++i)
            for (nn::Index j = 0; j < orig.cols(); ++j)
                EXPECT_FLOAT_EQ(orig.at(i, j), got.at(i, j));
    }

    std::filesystem::remove(path);
    std::filesystem::remove(tmpdir);
}
