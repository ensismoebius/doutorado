#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>

#include "core/dataLoaders/MatFileUtils.h"

using namespace matioCpp;

TEST(MatFileRowVectorTest, LoadRowVector)
{
    std::filesystem::remove("rowvec_test.mat");

    // Create a 1x4 row vector (stored column-major as 1x4)
    std::vector<double> raw = {1.0, 2.0, 3.0, 4.0};
    File file = File::Create("rowvec_test.mat");
    MultiDimensionalArray<double> rv("rvec", {1, 4}, raw.data());
    file.write(rv);
    file.close();

    auto opt = matioCpp::utils::load_named_variable_as_matrix("rowvec_test.mat", "rvec");
    ASSERT_TRUE(opt.has_value());
    auto m = opt.value();
    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 4);
    EXPECT_FLOAT_EQ(m(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(m(0, 3), 4.0F);
}
