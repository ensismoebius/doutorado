#include "MatFileUtils.h"

#include <iostream>

namespace matio::utils
{
static auto build_matrix_from_matvar(const MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    // Determine shape
    int rows = 1;
    int cols = 1;
    if (!mat_var.dimensions.empty())
    {
        rows = mat_var.dimensions.size() >= 1 ? mat_var.dimensions[0] : 1;
        cols = mat_var.dimensions.size() >= 2 ? mat_var.dimensions[1] : 1;
    }

    // Guard against non-positive dims
    if (rows <= 0 || cols <= 0)
    {
        return std::nullopt;
    }

    Eigen::MatrixXf mat(rows, cols);
    mat.setZero();

    try
    {
        if (mat_var.holds_type<double>())
        {
            const auto& vec = mat_var.get_vector<double>();
            size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            size_t copy_len = std::min(expected, vec.size());
            for (size_t i = 0; i < copy_len; ++i)
            {
                int r = static_cast<int>(i % static_cast<size_t>(rows));
                int c = static_cast<int>(i / static_cast<size_t>(rows));
                mat(r, c) = static_cast<float>(vec[i]);
            }
        }
        else if (mat_var.holds_type<float>())
        {
            const auto& vec = mat_var.get_vector<float>();
            size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            size_t copy_len = std::min(expected, vec.size());
            for (size_t i = 0; i < copy_len; ++i)
            {
                int r = static_cast<int>(i % static_cast<size_t>(rows));
                int c = static_cast<int>(i / static_cast<size_t>(rows));
                mat(r, c) = vec[i];
            }
        }
        else if (mat_var.holds_type<int32_t>())
        {
            const auto& vec = mat_var.get_vector<int32_t>();
            size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
            size_t copy_len = std::min(expected, vec.size());
            for (size_t i = 0; i < copy_len; ++i)
            {
                int r = static_cast<int>(i % static_cast<size_t>(rows));
                int c = static_cast<int>(i / static_cast<size_t>(rows));
                mat(r, c) = static_cast<float>(vec[i]);
            }
        }
        else
        {
            // Unsupported type
            return std::nullopt;
        }
    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Allocation error while constructing Eigen matrix: " << e.what() << '\n';
        return std::nullopt;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error while reading matvar data: " << e.what() << '\n';
        return std::nullopt;
    }

    return mat;
}

auto load_named_variable_as_matrix(const std::string& mat_path, const std::string& var_name)
    -> std::optional<Eigen::MatrixXf>
{
    MatFile mf;
    if (!mf.open(mat_path))
    {
        std::cerr << "Failed to open MAT file: " << mat_path << '\n';
        return std::nullopt;
    }

    auto var_opt = mf.read_variable(var_name);
    if (!var_opt)
    {
        std::cerr << "Variable '" << var_name << "' not found in file: " << mat_path << '\n';
        return std::nullopt;
    }

    const MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils
