#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>

#include "core/dataLoaders/mat_file_utils.hpp"

using namespace matioCpp;

TEST(MatFileVectorTest, LoadColumnVector)
{
    std::filesystem::remove("vec_test.mat");

    // Create a 4x1 column vector
    std::vector<double> raw = {10.0, 20.0, 30.0, 40.0};
    File file = File::Create("vec_test.mat");
    MultiDimensionalArray<double> vec("vcol", {4, 1}, raw.data());
    file.write(vec);
    file.close();

    auto opt = matioCpp::utils::load_named_variable_as_matrix("vec_test.mat", "vcol");
    ASSERT_TRUE(opt.has_value());
    auto m = opt.value();
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_FLOAT_EQ(m(0, 0), 10.0F);
    EXPECT_FLOAT_EQ(m(3, 0), 40.0F);
}
