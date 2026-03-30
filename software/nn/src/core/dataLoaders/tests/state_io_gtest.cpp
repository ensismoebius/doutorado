#include <gtest/gtest.h>

#include "nn/serialization/StateIO.hpp"
#include "nn/testing/tempfile.hpp"
#include <cstdio>

TEST(StateIO, RoundtripMap)
{
    std::map<std::string, nn::Tensor> m;
    nn::Tensor a(2, 3);
    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j) a.at(i, j) = static_cast<float>(i * 10 + j + 1);
    m["weights"] = a;
    nn::Tensor b(1, 1);
    b.at(0, 0) = 3.1415f;
    m["scalar"] = b;

    const std::string tmp = nn::testing::make_temp_file("stateio_");
    ASSERT_FALSE(tmp.empty());
    bool ok = nn::serialization::save_state_dict(m, tmp);
    ASSERT_TRUE(ok);

    auto loaded = nn::serialization::load_state_dict(tmp);
    ASSERT_EQ(loaded.size(), m.size());
    ASSERT_TRUE(loaded.count("weights"));
    ASSERT_TRUE(loaded.count("scalar"));
    auto lw = loaded["weights"];
    ASSERT_EQ(lw.rows(), a.rows());
    ASSERT_EQ(lw.cols(), a.cols());
    for (size_t i = 0; i < lw.rows(); ++i)
        for (size_t j = 0; j < lw.cols(); ++j) ASSERT_FLOAT_EQ(lw.at(i, j), a.at(i, j));

    auto ls = loaded["scalar"];
    ASSERT_EQ(ls.rows(), 1);
    ASSERT_EQ(ls.cols(), 1);
    ASSERT_FLOAT_EQ(ls.at(0, 0), b.at(0, 0));

    std::remove(tmp.c_str());
}
