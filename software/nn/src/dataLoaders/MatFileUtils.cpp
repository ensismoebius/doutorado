#include "MatFileUtils.h"

#include <matioCpp/EigenConversions.h>
#include <matioCpp/File.h>

namespace matioCpp::utils
{

auto load_named_variable_as_matrix(const std::string& mat_path, const std::string& var_name)
    -> std::optional<Eigen::MatrixXf>
{
    try
    {
        matioCpp::File file(mat_path);
        auto variable = file.read(var_name);
        if (!variable.isValid()) return std::nullopt;

        // Accept both 2-D arrays and vectors (column vectors) from matio-cpp
        if (variable.variableType() != matioCpp::VariableType::MultiDimensionalArray &&
            variable.variableType() != matioCpp::VariableType::Vector)
        {
            // diagnostic
            std::cerr << "MatFileUtils: variable '" << var_name
                      << "' is not a MultiDimensionalArray or Vector\n";
            return std::nullopt;
        }

        // Handle the actual stored ValueType safely. matio-cpp will assert if
        // you try to cast a non-convertible stored type to the requested template
        // parameter in asMultiDimensionalArray<T>(). Read with the correct
        // template type and then convert to float matrix.
        using ValueType = matioCpp::ValueType;
        const auto vt = variable.valueType();

        if (variable.isComplex())
        {
            // Complex arrays are not supported by this utility yet.
            std::cerr << "MatFileUtils: variable '" << var_name << "' is complex and unsupported\n";
            return std::nullopt;
        }

        switch (vt)
        {
            case ValueType::DOUBLE:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<double>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<double>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::SINGLE:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<float>();
                    return matioCpp::to_eigen(mat);
                }
                else
                {
                    auto vec = variable.asVector<float>();
                    return matioCpp::to_eigen(vec);
                }
            }
            case ValueType::INT8:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<int8_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<int8_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::UINT8:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<uint8_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<uint8_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::INT16:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<int16_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<int16_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::UINT16:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<uint16_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<uint16_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::INT32:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<int32_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<int32_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::UINT32:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<uint32_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<uint32_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::INT64:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<int64_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<int64_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            case ValueType::UINT64:
            {
                if (variable.variableType() == matioCpp::VariableType::MultiDimensionalArray)
                {
                    auto mat = variable.asMultiDimensionalArray<uint64_t>();
                    return matioCpp::to_eigen(mat).cast<float>();
                }
                else
                {
                    auto vec = variable.asVector<uint64_t>();
                    return matioCpp::to_eigen(vec).template cast<float>();
                }
            }
            default:
                // Unsupported or non-numeric types (strings, logical, etc.)
                return std::nullopt;
        }
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

auto list_variable_names(const std::string& mat_path) -> std::vector<std::string>
{
    try
    {
        matioCpp::File file(mat_path);
        return file.variableNames();
    }
    catch (const std::exception&)
    {
        return {};
    }
}

} // namespace matioCpp::utils