/**
 * @file matfile_utils_gtest.cpp
 * @brief Unit tests for MAT-file utility helpers (matioCpp -> Eigen/Tensor mapping).
 */

#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <algorithm>
#include <filesystem>
#include <vector>

#include "nn/dataLoaders/io/mat_file_utils.hpp"

using namespace matioCpp;

TEST(MatFileUtilsTest, LoadExistingDoubleMatrix)
{
    const auto mat_path = std::filesystem::temp_directory_path() / "utils_test.mat";
    std::filesystem::remove(mat_path);
    // Prepare column-major data: [1,3,2,4] corresponds to [[1,2],[3,4]]
    std::vector<double> raw = {1.0, 3.0, 2.0, 4.0};
    File file = File::Create(mat_path.string());
    MultiDimensionalArray<double> dmat("dmat", {2, 2}, raw.data());
    file.write(dmat);

    auto mat_opt = utils::load_named_variable_as_matrix(mat_path.string(), "dmat");
    ASSERT_TRUE(mat_opt.has_value());
    auto mat = mat_opt.value();
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
    EXPECT_FLOAT_EQ(mat.at(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(mat.at(1, 1), 4.0F);
}

TEST(MatFileUtilsTest, MissingVariableReturnsNullopt)
{
    const auto mat_path = std::filesystem::temp_directory_path() / "utils_test2.mat";
    std::filesystem::remove(mat_path);
    File file = File::Create(mat_path.string());
    file.close();

    auto mat_opt = utils::load_named_variable_as_matrix(mat_path.string(), "nope");
    EXPECT_FALSE(mat_opt.has_value());
}

TEST(MatFileUtilsTest, LoadVariousIntegerTypes)
{
    const auto mat_path = std::filesystem::temp_directory_path() / "utils_test3.mat";
    std::filesystem::remove(mat_path);
    File file = File::Create(mat_path.string());
    std::vector<int32_t> raw1 = {1, 3, 2, 4};
    MultiDimensionalArray<int32_t> i32("i32", {2, 2}, raw1.data());
    file.write(i32);

    std::vector<int32_t> raw2 = {-1, -3, -2, -4};
    MultiDimensionalArray<int32_t> i32b("i32b", {2, 2}, raw2.data());
    file.write(i32b);

    auto m1 = utils::load_named_variable_as_matrix(mat_path.string(), "i32");
    ASSERT_TRUE(m1.has_value());
    // Calculate sum manually: 1+3+2+4 = 10
    float sum = 0.0F;
    for (int r = 0; r < m1->rows(); ++r)
    {
        for (int c = 0; c < m1->cols(); ++c)
        {
            sum += m1->at(r, c);
        }
    }
    EXPECT_FLOAT_EQ(sum, 10.0F);

    auto m2 = utils::load_named_variable_as_matrix(mat_path.string(), "i32b");
    ASSERT_TRUE(m2.has_value());
    EXPECT_FLOAT_EQ(m2->at(0, 0), -1.0F);
}

TEST(MatFileUtilsTest, ListVariableNamesReturnsAllTopLevelVariables)
{
    const auto mat_path = std::filesystem::temp_directory_path() / "utils_list_names_test.mat";
    std::filesystem::remove(mat_path);

    std::vector<double> a_data = {1.0, 3.0, 2.0, 4.0}; // 2x2, column-major
    std::vector<double> b_data = {10.0, 20.0};         // 2x1

    File file = File::Create(mat_path.string());
    MultiDimensionalArray<double> a_var("A", {2, 2}, a_data.data());
    MultiDimensionalArray<double> b_var("B", {2, 1}, b_data.data());
    file.write(a_var);
    file.write(b_var);
    file.close();

    auto names = utils::list_variable_names(mat_path.string());
    ASSERT_EQ(names.size(), 2U);

    EXPECT_NE(std::find(names.begin(), names.end(), "A"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "B"), names.end());

    std::filesystem::remove(mat_path);
}