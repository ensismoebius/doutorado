/**
 * @file MatTestUtils.cpp
 * @brief Helpers for writing small MAT v5 test fixtures using matio.
 */

#include "MatTestUtils.h"

#include <matio.h>

namespace nn::dataLoaders::test
{
auto writeMatDouble(const std::string& filePath,
    const std::string& varName,
    size_t rows,
    size_t cols,
    const double* data) noexcept -> bool
{
    mat_t* mat = Mat_CreateVer(filePath.c_str(), nullptr, MAT_FT_MAT5);
    if (mat == nullptr)
    {
        return false; // LCOV_EXCL_LINE
    }

    size_t dims[2];
    dims[0] = rows;
    dims[1] = cols;
    matvar_t* matvar = Mat_VarCreate(
        varName.c_str(), MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, const_cast<double*>(data), 0);
    if (matvar == nullptr)
    {
        Mat_Close(mat); // LCOV_EXCL_LINE
        return false;   // LCOV_EXCL_LINE
    }
    int ok = Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);
    Mat_Close(mat);
    return ok == 0;
}

auto writeMatFloat(const std::string& filePath,
    const std::string& varName,
    size_t rows,
    size_t cols,
    const float* data) noexcept -> bool
{
    mat_t* mat = Mat_CreateVer(filePath.c_str(), nullptr, MAT_FT_MAT5);
    if (mat == nullptr)
    {
        return false;
    }
    size_t dims[2]; // LCOV_EXCL_LINE
    dims[0] = rows;
    dims[1] = cols;
    matvar_t* matvar = Mat_VarCreate(
        varName.c_str(), MAT_C_SINGLE, MAT_T_SINGLE, 2, dims, const_cast<float*>(data), 0);
    if (matvar == nullptr)
    {
        Mat_Close(mat); // LCOV_EXCL_LINE
        return false;   // LCOV_EXCL_LINE
    }
    int ok = Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE);
    Mat_VarFree(matvar);
    Mat_Close(mat);
    return ok == 0;
}

} // namespace nn::dataLoaders::test
