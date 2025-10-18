#include <iostream>
#include <vector>

#include "dataLoaders/MatFile.h"

using namespace matio;

auto main() -> int {
  // Create a new MAT file
  MatFile mat_file;
  if (!mat_file.create("example.mat")) {
    std::cerr << "Failed to create MAT file\n";
    return 1;
  }

  // Write some variables
  std::vector<double> matrix_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Dimensions matrix_dims = {2, 3};  // 2x3 matrix

  if (!mat_file.write_double_matrix(
          "double_matrix", matrix_data, matrix_dims)) {
    std::cerr << "Failed to write double matrix\n";
  }

  std::vector<int32_t> int_data = {10, 20, 30, 40};
  if (!mat_file.write_int32_matrix("int_matrix", int_data, {2, 2})) {
    std::cerr << "Failed to write int matrix\n";
  }

  if (!mat_file.write_string("message", "Hello from C++20!")) {
    std::cerr << "Failed to write string\n";
  }

  mat_file.close();

  // Read the file back
  if (!mat_file.open("example.mat")) {
    std::cerr << "Failed to open MAT file\n";
    return 1;
  }

  auto variables = mat_file.read_all_variables();

  std::cout << "Variables in file:\n";
  for (const auto& [name, var] : variables) {
    std::cout << "  " << name << ": " << var.type_name() << " ["
              << var.dimensions[0] << "x" << var.dimensions[1] << "]\n";

    if (var.holds_type<double>()) {
      const auto& data = var.get_vector<double>();
      std::cout << "    Data: ";
      for (size_t i = 0; i < std::min(data.size(), size_t{5}); ++i) {
        std::cout << data[i] << " ";
      }
      std::cout << "\n";
    } else if (var.holds_type<std::string>()) {
      const auto& str = std::get<std::string>(var.data);
      std::cout << "    Data: " << str << "\n";
    }
  }

  return 0;
}