#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>
#include <vector>

#include "dataLoaders/MatFileUtils.h"

using namespace matioCpp;

TEST(MatFileUtilsTest, LoadExistingDoubleMatrix)
{
    std::filesystem::remove("utils_test.mat");
    // Prepare column-major data: [1,3,2,4] corresponds to [[1,2],[3,4]]
    std::vector<double> raw = {1.0, 3.0, 2.0, 4.0};
    File file = File::Create("utils_test.mat");
    MultiDimensionalArray<double> dmat("dmat", {2, 2}, raw.data());
    file.write(dmat);

    auto mat_opt = utils::load_named_variable_as_matrix("utils_test.mat", "dmat");
    ASSERT_TRUE(mat_opt.has_value());
    auto mat = mat_opt.value();
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
    EXPECT_FLOAT_EQ(mat(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(mat(1, 1), 4.0F);
}

TEST(MatFileUtilsTest, MissingVariableReturnsNullopt)
{
    std::filesystem::remove("utils_test2.mat");
    File file = File::Create("utils_test2.mat");
    file.close();

    auto mat_opt = utils::load_named_variable_as_matrix("utils_test2.mat", "nope");
    EXPECT_FALSE(mat_opt.has_value());
}

TEST(MatFileUtilsTest, LoadVariousIntegerTypes)
{
    std::filesystem::remove("utils_test3.mat");
    File file = File::Create("utils_test3.mat");
    std::vector<int32_t> raw1 = {1, 3, 2, 4};
    MultiDimensionalArray<int32_t> i32("i32", {2, 2}, raw1.data());
    file.write(i32);

    std::vector<int32_t> raw2 = {-1, -3, -2, -4};
    MultiDimensionalArray<int32_t> i32b("i32b", {2, 2}, raw2.data());
    file.write(i32b);

    auto m1 = utils::load_named_variable_as_matrix("utils_test3.mat", "i32");
    ASSERT_TRUE(m1.has_value());
    EXPECT_FLOAT_EQ(m1->sum(), 10.0F);

    auto m2 = utils::load_named_variable_as_matrix("utils_test3.mat", "i32b");
    ASSERT_TRUE(m2.has_value());
    EXPECT_FLOAT_EQ((*m2)(0, 0), -1.0F);
}