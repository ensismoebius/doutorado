// MatFileUtils_impl.cpp - clean, compact implementation
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "dataLoaders/MatFile.h"
#include "dataLoaders/MatFileUtils.h"

namespace matio::utils
{

template <typename T>
static bool copy_if_type(const matio::MatVar& var, Eigen::MatrixXf& out, int rows, int cols)
{
    if (!var.holds_type<T>()) return false;
    const auto& v = var.get_vector<T>();
    const size_t N = v.size();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            const size_t idx =
                static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
            if (idx < N)
                out(r, c) = static_cast<float>(v[idx]);
            else
                out(r, c) = 0.0f;
        }
    }
    return true;
}

static auto build_matrix_from_matvar(const matio::MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    int rows = 1, cols = 1;
    if (!mat_var.dimensions.empty())
    {
        rows = static_cast<int>(mat_var.dimensions.size() >= 1 ? mat_var.dimensions[0] : 1);
        cols = static_cast<int>(mat_var.dimensions.size() >= 2 ? mat_var.dimensions[1] : 1);
    }
    if (rows <= 0 || cols <= 0) return std::nullopt;

    Eigen::MatrixXf mat(rows, cols);
    mat.setZero();

    try
    {
        // try supported types; order: most common numeric types first
        if (copy_if_type<double>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<float>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<int32_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<int64_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<uint32_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<uint64_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<int16_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<uint16_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<int8_t>(mat_var, mat, rows, cols)) return mat;
        if (copy_if_type<uint8_t>(mat_var, mat, rows, cols)) return mat;

        // unsupported type
        return std::nullopt;
    }
    catch (const std::bad_alloc& e)
    {
        std::cerr << "Allocation error in build_matrix_from_matvar: " << e.what() << '\n';
        return std::nullopt;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in build_matrix_from_matvar: " << e.what() << '\n';
        return std::nullopt;
    }
}

auto load_named_variable_as_matrix(const std::string& mat_path, const std::string& var_name)
    -> std::optional<Eigen::MatrixXf>
{
    matio::MatFile mf;
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

    return build_matrix_from_matvar(var_opt.value());
}

} // namespace matio::utils
