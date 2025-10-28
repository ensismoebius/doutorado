#include <gtest/gtest.h>

#include <filesystem>

#include "dataLoaders/MatFile.h"
#include "dataLoaders/MatFileUtils.h"

using namespace matio;

TEST(MatFileUtilsTest, LoadExistingDoubleMatrix)
{
    std::filesystem::remove("utils_test.mat");
    MatFile mf;
    ASSERT_TRUE(mf.create("utils_test.mat"));
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
    mf.write_double_matrix("dmat", data, {2, 2});
    mf.close();

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
    MatFile mf;
    ASSERT_TRUE(mf.create("utils_test2.mat"));
    mf.close();

    auto mat_opt = utils::load_named_variable_as_matrix("utils_test2.mat", "nope");
    EXPECT_FALSE(mat_opt.has_value());
}

TEST(MatFileUtilsTest, LoadVariousIntegerTypes)
{
    std::filesystem::remove("utils_test3.mat");
    MatFile mf;
    ASSERT_TRUE(mf.create("utils_test3.mat"));

    mf.write_int32_matrix("i32", {1, 2, 3, 4}, {2, 2});
    mf.write_int32_matrix("i32b", {-1, -2, -3, -4}, {2, 2});
    mf.close();

    auto m1 = utils::load_named_variable_as_matrix("utils_test3.mat", "i32");
    ASSERT_TRUE(m1.has_value());
    EXPECT_FLOAT_EQ(m1->sum(), 10.0F);

    auto m2 = utils::load_named_variable_as_matrix("utils_test3.mat", "i32b");
    ASSERT_TRUE(m2.has_value());
    EXPECT_FLOAT_EQ((*m2)(0, 0), -1.0F);
}
