#ifndef NN_DATALOADERS_MAT_FILE_UTILS_HPP
#define NN_DATALOADERS_MAT_FILE_UTILS_HPP

#include <optional>
#include <string>
#include <vector>

#include "nn/tensor/Tensor.hpp"

namespace matioCpp::utils
{
/**
 * @file mat_file_utils.hpp
 * @brief Convenience helpers for loading MATLAB variables into `nn::Tensor`.
 *
 * Design goals:
 * - Keep demo/experiment code simple by providing "load by name" building blocks.
 * - Fail safely: functions return `std::nullopt` / empty vectors rather than
 *   throwing, so callers can decide how to report errors.
 *
 * Mapping notes:
 * - The loader targets numeric matrices that can be represented as float data in
 *   this project's `nn::Tensor`.
 * - Higher-dimensional arrays may be rejected unless explicitly supported.
 */

// Load a named variable from a .mat file and return it as an nn::Tensor.
// Returns std::nullopt on any error (file can't be opened, variable not
// found, unsupported type, or dimension mismatch).
[[nodiscard]] auto load_named_variable_as_matrix(const std::string& mat_path,
                                                 const std::string& var_name)
    -> std::optional<nn::Tensor>;

// Return a list of top-level variable names available in the .mat file.
// Returns an empty vector on error (file can't be opened or parsing failed).
[[nodiscard]] auto list_variable_names(const std::string& mat_path) -> std::vector<std::string>;

// Get the dimensions of a named variable from a .mat file without loading its full content.
// Returns std::nullopt on error (file can't be opened, variable not found).
[[nodiscard]] auto get_variable_dimensions(const std::string& mat_path, const std::string& var_name)
    -> std::optional<std::vector<size_t>>;

// Example:
// auto mat = matio::utils::load_named_variable_as_matrix("/tmp/file.mat", "data");
// if (mat) { std::cout << "Loaded matrix: " << mat->rows() << "x" << mat->cols() << '\n'; }

} // namespace matioCpp::utils
#endif // NN_DATALOADERS_MAT_FILE_UTILS_HPP
