#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

namespace matioCpp::utils
{
// Load a named variable from a .mat file and return it as an Eigen::MatrixXf.
// Returns std::nullopt on any error (file can't be opened, variable not
// found, unsupported type, or dimension mismatch).
[[nodiscard]] auto load_named_variable_as_matrix(const std::string& mat_path,
                                                 const std::string& var_name)
    -> std::optional<Eigen::MatrixXf>;

// Return a list of top-level variable names available in the .mat file.
// Returns an empty vector on error (file can't be opened or parsing failed).
[[nodiscard]] auto list_variable_names(const std::string& mat_path) -> std::vector<std::string>;

// Get the dimensions of a named variable from a .mat file without loading its full content.
// Returns std::nullopt on error (file can't be opened, variable not found).
[[nodiscard]] auto get_variable_dimensions(const std::string& mat_path,
                                           const std::string& var_name)
    -> std::optional<std::vector<size_t>>;

// Example:
// auto mat = matio::utils::load_named_variable_as_matrix("/tmp/file.mat", "data");
// if (mat) { std::cout << "Loaded matrix: " << mat->rows() << "x" << mat->cols() << '\n'; }

} // namespace matioCpp::utils