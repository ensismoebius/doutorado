/**
 * @file mat_file_utils.cpp
 * @brief Conversions from matio-cpp variable types into `nn::Tensor`.
 */

#include "nn/dataLoaders/mat_file_utils.hpp"

#include <matio.h>
#include <matioCpp/EigenConversions.h>
#include <matioCpp/File.h>

#include <algorithm>
#include <stdexcept>

#include "nn/dataLoaders/mat_file.hpp"

// Implementation strategy:
// - matio-cpp reads variables and can convert them to Eigen structures.
// - This translation unit converts supported variable types into `nn::Tensor`
//   by copying into the project's float-based tensor storage.
//
// Supported forms:
// - Multi-dimensional arrays and vectors
// - Real (non-complex) numeric types (double/float and integral types)
//
// Failure model:
// - Helpers return `std::nullopt` on unsupported types, missing variables, or
//   any exception thrown by the underlying matio-cpp library.

namespace matioCpp::utils
{
namespace
{
auto to_tensor_from_eigen_matrix(const Eigen::MatrixXf& eigen_matrix) -> nn::Tensor
{
    nn::Tensor result(
        static_cast<size_t>(eigen_matrix.rows()), static_cast<size_t>(eigen_matrix.cols()));
    std::copy_n(
        eigen_matrix.data(), static_cast<size_t>(eigen_matrix.size()), result.mutable_data_ptr());
    return result;
}
} // namespace

// Small helpers that convert matio-cpp Variable objects into nn::Tensor instances without
// exposing Eigen types in this translation unit.
template <typename T>
auto to_tensor_from_multi(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto multi_array = variable.template asMultiDimensionalArray<T>();
    auto eigen_matrix = matioCpp::to_eigen(multi_array).template cast<float>();
    return to_tensor_from_eigen_matrix(eigen_matrix);
}

template <typename T>
auto to_tensor_from_vector(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto vec = variable.template asVector<T>();
    auto eigen_matrix = matioCpp::to_eigen(vec).template cast<float>();
    return to_tensor_from_eigen_matrix(eigen_matrix);
}

auto load_named_variable_as_matrix(const std::string& mat_path, const std::string& var_name)
    -> std::optional<nn::Tensor>
{
    try
    {
        matioCpp::File file(mat_path);
        auto variable = file.read(var_name); // flawfinder: ignore
        if (!variable.isValid())
        {
            return std::nullopt;
        }

        bool is_multi = variable.variableType() == matioCpp::VariableType::MultiDimensionalArray;
        bool is_vector = variable.variableType() == matioCpp::VariableType::Vector;
        if (!is_multi && !is_vector)
        {
            return std::nullopt;
        }

        if (variable.isComplex())
        {
            return std::nullopt;
        }

        using ValueType = matioCpp::ValueType;
        const auto vt = variable.valueType();

        switch (vt)
        {
            case ValueType::DOUBLE:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<double>(variable)
                           : to_tensor_from_vector<double>(variable);
            case ValueType::SINGLE:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<float>(variable)
                           : to_tensor_from_vector<float>(variable);
            case ValueType::INT8:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<int8_t>(variable)
                           : to_tensor_from_vector<int8_t>(variable);
            case ValueType::UINT8:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<uint8_t>(variable)
                           : to_tensor_from_vector<uint8_t>(variable);
            case ValueType::INT16:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<int16_t>(variable)
                           : to_tensor_from_vector<int16_t>(variable);
            case ValueType::UINT16:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<uint16_t>(variable)
                           : to_tensor_from_vector<uint16_t>(variable);
            case ValueType::INT32:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<int32_t>(variable)
                           : to_tensor_from_vector<int32_t>(variable);
            case ValueType::UINT32:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<uint32_t>(variable)
                           : to_tensor_from_vector<uint32_t>(variable);
            case ValueType::INT64:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<int64_t>(variable)
                           : to_tensor_from_vector<int64_t>(variable);
            case ValueType::UINT64:
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<uint64_t>(variable)
                           : to_tensor_from_vector<uint64_t>(variable);
            default:
                return std::nullopt;
        }
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

auto countMatRows(const std::string& matPath, const std::string& varName) -> std::size_t
{
    auto dims = get_variable_dimensions(matPath, varName);
    if (!dims)
    {
        throw std::runtime_error(
            "Failed to read dimensions for variable '" + varName + "' in " + matPath);
    }
    return (*dims)[0];
}

auto get_variable_dimensions(const std::string& mat_path, const std::string& var_name)
    -> std::optional<std::vector<size_t>>
{
    mat_t* mat_file = Mat_Open(mat_path.c_str(), MAT_ACC_RDONLY);
    if (mat_file == nullptr)
    {
        return std::nullopt;
    }

    matvar_t* var_info = Mat_VarReadInfo(mat_file, var_name.c_str());
    if (var_info == nullptr)
    {
        Mat_Close(mat_file);
        return std::nullopt;
    }

    if (var_info->dims == nullptr || var_info->rank <= 0)
    {
        Mat_VarFree(var_info);
        Mat_Close(mat_file);
        return std::nullopt;
    }

    std::vector<size_t> dims(static_cast<size_t>(var_info->rank));
    for (int i = 0; i < var_info->rank; ++i)
    {
        dims[static_cast<size_t>(i)] = static_cast<size_t>(var_info->dims[i]);
    }

    Mat_VarFree(var_info);
    Mat_Close(mat_file);
    return dims;
}

[[nodiscard]] auto list_variable_names(const std::string& mat_path) -> std::vector<std::string>
{
    mat_t* mat_file = Mat_Open(mat_path.c_str(), MAT_ACC_RDONLY);
    if (mat_file == nullptr)
    {
        return {};
    }

    size_t var_count = 0;
    char** dir = Mat_GetDir(mat_file, &var_count);
    if (dir == nullptr)
    {
        Mat_Close(mat_file);
        return {};
    }

    std::vector<std::string> names;
    names.reserve(var_count);
    for (size_t i = 0; i < var_count; ++i)
    {
        if (dir[i] != nullptr)
        {
            names.emplace_back(dir[i]);
        }
    }

    Mat_Close(mat_file);
    return names;
}

} // namespace matioCpp::utils
