#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>

namespace matio::utils
{
// Load a named variable from a .mat file and return it as an Eigen::MatrixXf.
// Returns std::nullopt on any error (file can't be opened, variable not
// found, unsupported type, or dimension mismatch).
[[nodiscard]] auto load_named_variable_as_matrix(const std::string& mat_path,
                                                 const std::string& var_name)
    -> std::optional<Eigen::MatrixXf>;

// Example:
// auto mat = matio::utils::load_named_variable_as_matrix("/tmp/file.mat", "data");
// if (mat) { std::cout << "Loaded matrix: " << mat->rows() << "x" << mat->cols() << '\n'; }

} // namespace matio::utils
