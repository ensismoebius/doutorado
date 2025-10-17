#include <gtest/gtest.h>

#include <vector>

#include "mat_file.h"

using namespace matio;

TEST(MatFileTest, CreateAndWriteReadFile) {
  MatFile mat_file;
  ASSERT_TRUE(mat_file.create("test.mat"));

  std::vector<double> double_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Dimensions double_dims = {2, 3};
  ASSERT_TRUE(
      mat_file.write_double_matrix("double_matrix", double_data, double_dims));

  std::vector<float> float_data = {1.0F, 2.0F, 3.0F, 4.0F};
  Dimensions float_dims = {2, 2};
  ASSERT_TRUE(
      mat_file.write_single_matrix("float_matrix", float_data, float_dims));

  std::vector<int32_t> int_data = {10, 20, 30, 40};
  Dimensions int_dims = {2, 2};
  ASSERT_TRUE(mat_file.write_int32_matrix("int_matrix", int_data, int_dims));

  std::string str_data = "Hello, MAT file!";
  ASSERT_TRUE(mat_file.write_string("string_data", str_data));

  mat_file.close();

  MatFile read_mat_file;
  ASSERT_TRUE(read_mat_file.open("test.mat"));

  auto variables = read_mat_file.read_all_variables();

  // Check double matrix
  auto double_var_it = variables.find("double_matrix");
  ASSERT_NE(double_var_it, variables.end());
  const auto& double_var = double_var_it->second;
  ASSERT_TRUE(double_var.holds_type<double>());
  const auto& read_double_data = double_var.get_vector<double>();
  ASSERT_EQ(read_double_data.size(), double_data.size());
  for (size_t i = 0; i < double_data.size(); ++i) {
    EXPECT_EQ(double_data[i], read_double_data[i]);
  }

  // Check float matrix
  auto float_var_it = variables.find("float_matrix");
  ASSERT_NE(float_var_it, variables.end());
  const auto& float_var = float_var_it->second;
  ASSERT_TRUE(float_var.holds_type<float>());
  const auto& read_float_data = float_var.get_vector<float>();
  ASSERT_EQ(read_float_data.size(), float_data.size());
  for (size_t i = 0; i < float_data.size(); ++i) {
    EXPECT_EQ(float_data[i], read_float_data[i]);
  }

  // Check int matrix
  auto int_var_it = variables.find("int_matrix");
  ASSERT_NE(int_var_it, variables.end());
  const auto& int_var = int_var_it->second;
  ASSERT_TRUE(int_var.holds_type<int32_t>());
  const auto& read_int_data = int_var.get_vector<int32_t>();
  ASSERT_EQ(read_int_data.size(), int_data.size());
  for (size_t i = 0; i < int_data.size(); ++i) {
    EXPECT_EQ(int_data[i], read_int_data[i]);
  }

  // Check string
  auto string_var_it = variables.find("string_data");
  ASSERT_NE(string_var_it, variables.end());
  const auto& string_var = string_var_it->second;
  ASSERT_TRUE(string_var.holds_type<std::string>());
  const auto& read_str_data = std::get<std::string>(string_var.data);
  ASSERT_EQ(read_str_data, str_data);
}
