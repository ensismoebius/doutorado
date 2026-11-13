/**
 * @file mat_file_utils.cpp
 * @brief Conversions from matio-cpp variable types into `nn::Tensor`.
 */

#include "data_loaders/mat_io/mat_file_utils.hpp"

#include <matio.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>
#include <matioCpp/Vector.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

#include "data_loaders/mat_io/mat_file.hpp"
#include "nlohmann/json.hpp"

// Implementation strategy:
// - matio-cpp reads variables and can convert them to raw data pointers.
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
template <typename T>
auto to_tensor_from_raw(const T* data, size_t rows, size_t cols) -> nn::Tensor
{
    nn::Tensor result(rows, cols);
    // MATLAB/matio stores column-major; use backend-agnostic at() for safe writes.
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c)
            result.at(static_cast<nn::Index>(r), static_cast<nn::Index>(c)) =
                static_cast<float>(data[c * rows + r]);
    return result;
}
} // namespace

// Small helpers that convert matio-cpp Variable objects into nn::Tensor instances.
template <typename T>
auto to_tensor_from_multi(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto multi = variable.template asMultiDimensionalArray<T>();
    auto dims = multi.dimensions();
    size_t rows = (dims.size() > 0) ? dims[0] : 1;
    size_t cols = (dims.size() > 1) ? dims[1] : 1;
    return to_tensor_from_raw(multi.data(), rows, cols);
}

template <typename T>
auto to_tensor_from_vector(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto vec = variable.template asVector<T>();
    return to_tensor_from_raw(vec.data(), vec.size(), 1);
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
                // matioCpp exposes UINT64 data through its `size_t_type`
                // alias (uint64_t on Linux, unsigned long on macOS) rather
                // than raw uint64_t, so use it for portability.
                return (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                           ? to_tensor_from_multi<matioCpp::size_t_type>(variable)
                           : to_tensor_from_vector<matioCpp::size_t_type>(variable);
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

auto countShardRows(const std::string& shardIndexPath, const std::string& section) -> std::size_t
{
    // The shard index is JSON: {"<section>": [{"count": <rows>, ...}, ...]}. Every entry must
    // carry an integer count; a missing one is an error, not zero rows. (This used to run a
    // shell command with the path pasted into it and treat a missing count as 0.)
    std::ifstream in(shardIndexPath);
    if (!in)
    {
        throw std::runtime_error("Cannot open shard index: " + shardIndexPath);
    }
    nlohmann::json index;
    try
    {
        in >> index;
    }
    catch (const nlohmann::json::exception& e)
    {
        throw std::runtime_error(
            "Shard index is not valid JSON (" + std::string(e.what()) + "): " + shardIndexPath);
    }
    if (!index.is_object() || !index.contains(section) || !index.at(section).is_array())
    {
        throw std::runtime_error("Shard index has no '" + section + "' array: " + shardIndexPath);
    }
    std::size_t rows = 0;
    for (const auto& entry : index.at(section))
    {
        if (!entry.is_object() || !entry.contains("count") ||
            !entry.at("count").is_number_integer() || entry.at("count").get<long long>() < 0)
        {
            std::string msg = "Shard index entry in '";
            msg += section;
            msg += "' lacks a non-negative integer 'count': ";
            msg += shardIndexPath;
            throw std::runtime_error(msg);
        }
        rows += entry.at("count").get<std::size_t>();
    }
    return rows;
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
