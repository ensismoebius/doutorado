/**
 * @file MatTestUtils.h
 * @brief Small MAT-file creation helpers used by loader unit tests.
 */

// Small test utilities for creating MAT files used in unit tests
#pragma once

#include <matio.h>

#include <cstddef>
#include <string>

namespace nn::dataLoaders::test
{
// Create a MAT v5 file named `filePath` and write a 2D matrix variable `varName`
// with dimensions (rows x cols) using the provided pointer `data` (row-major as
// expected by matio: data array indexed as data[col*rows + row]). Returns true on success.
bool writeMatDouble(const std::string& filePath, const std::string& varName, size_t rows,
                    size_t cols, const double* data) noexcept;

// Create a MAT v5 file with single-precision data
bool writeMatFloat(const std::string& filePath, const std::string& varName, size_t rows,
                   size_t cols, const float* data) noexcept;

} // namespace nn::dataLoaders::test
