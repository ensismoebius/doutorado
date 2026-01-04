#include "mat_file_utils.hpp"

#include <matioCpp/EigenConversions.h>
#include <matioCpp/File.h>

namespace matioCpp::utils
{
namespace
{
// Small helpers that convert matio-cpp Variable objects into nn::Tensor instances without
// exposing Eigen types in this translation unit.
template <typename T>
auto to_tensor_from_multi(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto multi_array = variable.template asMultiDimensionalArray<T>();
    auto eigen_matrix = matioCpp::to_eigen(multi_array).template cast<float>();
    return nn::Tensor{eigen_matrix};
}

template <typename T>
auto to_tensor_from_vector(const matioCpp::Variable& variable) -> std::optional<nn::Tensor>
{
    auto vec = variable.template asVector<T>();
    auto eigen_matrix = matioCpp::to_eigen(vec).template cast<float>();
    return nn::Tensor{eigen_matrix};
}

} // anonymous namespace

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

} // namespace matioCpp::utils

namespace matioCpp::utils
{
auto get_variable_dimensions(const std::string& mat_path, const std::string& var_name)
    -> std::optional<std::vector<size_t>>
{
    try
    {
        matioCpp::File file(mat_path);
        auto variable = file.read(var_name); // flawfinder: ignore
        if (!variable.isValid())
        {
            return std::nullopt;
        }

        auto dims_span = variable.dimensions();
        return std::make_optional<std::vector<size_t>>(dims_span.begin(), dims_span.end());
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

[[nodiscard]] auto list_variable_names(const std::string& mat_path) -> std::vector<std::string>
{
    try
    {
        matioCpp::File file(mat_path);
        // matioCpp::File::variableNames() returns a std::vector<std::string>
        return file.variableNames();
    }
    catch (const std::exception&)
    {
        return {};
    }
}

} // namespace matioCpp::utils
