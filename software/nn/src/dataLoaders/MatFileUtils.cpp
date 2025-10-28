// MatFileUtils.cpp - minimal clean implementation
// Fully replaced with a single clean implementation
// Single, minimal implementation for MatFileUtils.cpp
#include "dataLoaders/MatFileUtils.h"

#include <cstdint>
#include <iostream>

#include "dataLoaders/MatFile.h"

namespace matio::utils
{

static auto build_matrix_from_matvar(const matio::MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    int rows = 1;
    int cols = 1;
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
        // only a compact set of supported types; copy in row-major order
        if (mat_var.holds_type<double>())
        {
            const auto& v = mat_var.get_vector<double>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<float>())
        {
            const auto& v = mat_var.get_vector<float>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = v[idx];
                }
        }
        else if (mat_var.holds_type<int32_t>())
        {
            const auto& v = mat_var.get_vector<int32_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else
        {
            // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils
for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c)
    {
        const size_t idx =
            static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
        if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
    }
}
else
{
    return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils
// Replace with minimal, single-definition implementation
#include <cstdint>
#include <iostream>

#include "dataLoaders/MatFile.h"
#include "dataLoaders/MatFileUtils.h"

namespace matio::utils
{

static auto build_matrix_from_matvar(const matio::MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    int rows = 1;
    int cols = 1;
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
        if (mat_var.holds_type<double>())
        {
            const auto& v = mat_var.get_vector<double>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<float>())
        {
            const auto& v = mat_var.get_vector<float>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = v[idx];
                }
        }
        else if (mat_var.holds_type<int32_t>())
        {
            const auto& v = mat_var.get_vector<int32_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<int8_t>())
        {
            const auto& v = mat_var.get_vector<int8_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<uint8_t>())
        {
            const auto& v = mat_var.get_vector<uint8_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<int16_t>())
        {
            const auto& v = mat_var.get_vector<int16_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<uint16_t>())
        {
            const auto& v = mat_var.get_vector<uint16_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<uint32_t>())
        {
            const auto& v = mat_var.get_vector<uint32_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<int64_t>())
        {
            const auto& v = mat_var.get_vector<int64_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else if (mat_var.holds_type<uint64_t>())
        {
            const auto& v = mat_var.get_vector<uint64_t>();
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < cols; ++c)
                {
                    const size_t idx =
                        static_cast<size_t>(r) * static_cast<size_t>(cols) + static_cast<size_t>(c);
                    if (idx < v.size()) mat(r, c) = static_cast<float>(v[idx]);
                }
        }
        else
        {
            return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils

// MatFileUtils.cpp — single clean implementation
#include <cstdint>
#include <iostream>

#include "MatFileUtils.h"
#include "dataLoaders/MatFile.h"

namespace matio
{
namespace utils
{

static auto build_matrix_from_matvar(const matio::MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    int rows = 1;
    int cols = 1;
    if (!mat_var.dimensions.empty())
    {
        rows = (mat_var.dimensions.size() >= 1) ? mat_var.dimensions[0] : 1;
        cols = (mat_var.dimensions.size() >= 2) ? mat_var.dimensions[1] : 1;
    }

    if (rows <= 0 || cols <= 0)
    {
        return std::nullopt;
    }

    Eigen::MatrixXf mat(rows, cols);
    mat.setZero();

// MatFileUtils.cpp — final clean implementation
#include <cstdint>
#include <iostream>

#include "MatFileUtils.h"

    namespace matio
    {
    namespace utils
    {

    static auto build_matrix_from_matvar(const matio::MatVar& mat_var)
        -> std::optional<Eigen::MatrixXf>
    {
        int rows = 1;
        int cols = 1;
        if (!mat_var.dimensions.empty())
        {
            rows = (mat_var.dimensions.size() >= 1) ? mat_var.dimensions[0] : 1;
            cols = (mat_var.dimensions.size() >= 2) ? mat_var.dimensions[1] : 1;
        }

        if (rows <= 0 || cols <= 0)
        {
            return std::nullopt;
        }

        Eigen::MatrixXf mat(rows, cols);
        mat.setZero();

// MatFileUtils.cpp — final clean implementation
#include <cstdint>
#include <iostream>

#include "MatFileUtils.h"

        namespace matio
        {
        namespace utils
        {

        static auto build_matrix_from_matvar(const MatVar& mat_var)
            -> std::optional<Eigen::MatrixXf>
        {
            int rows = 1;
            int cols = 1;
            if (!mat_var.dimensions.empty())
            {
                rows = (mat_var.dimensions.size() >= 1) ? mat_var.dimensions[0] : 1;
                cols = (mat_var.dimensions.size() >= 2) ? mat_var.dimensions[1] : 1;
            }

            if (rows <= 0 || cols <= 0)
            {
                return std::nullopt;
            }

            // Row-major copy: idx = r * cols + c
            mat.setZero();

            try
            {
                // Row-major copy: idx = r * cols + c
                if (mat_var.holds_type<double>())
                {
                    const auto& v = mat_var.get_vector<double>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<float>())
                {
                    const auto& v = mat_var.get_vector<float>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = v[idx];
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<int32_t>())
                {
                    const auto& v = mat_var.get_vector<int32_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<int8_t>())
                {
                    const auto& v = mat_var.get_vector<int8_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<uint8_t>())
                {
                    const auto& v = mat_var.get_vector<uint8_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<int16_t>())
                {
                    const auto& v = mat_var.get_vector<int16_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<uint16_t>())
                {
                    const auto& v = mat_var.get_vector<uint16_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<uint32_t>())
                {
                    const auto& v = mat_var.get_vector<uint32_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<int64_t>())
                {
                    const auto& v = mat_var.get_vector<int64_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else if (mat_var.holds_type<uint64_t>())
                {
                    const auto& v = mat_var.get_vector<uint64_t>();
                    for (int r = 0; r < rows; ++r)
                    {
                        for (int c = 0; c < cols; ++c)
                        {
                            size_t idx = static_cast<size_t>(r) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(c);
                            if (idx < v.size())
                            {
                                mat(r, c) = static_cast<float>(v[idx]);
                            }
                        }
                    }
                }
                else
                {
                    return std::nullopt; // unsupported type
                }
            }
            catch (const std::bad_alloc& e)
            {
                std::cerr << "Allocation error while constructing Eigen matrix: " << e.what()
                          << '\n';
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
            matio::MatFile mf;
            if (!mf.open(mat_path))
            {
                std::cerr << "Failed to open MAT file: " << mat_path << '\n';
                return std::nullopt;
            }

            auto var_opt = mf.read_variable(var_name);
            if (!var_opt)
            {
                std::cerr << "Variable '" << var_name << "' not found in file: " << mat_path
                          << '\n';
                return std::nullopt;
            }

            const matio::MatVar& var = var_opt.value();
            return build_matrix_from_matvar(var);
        }

        } // namespace utils
        } // namespace matio
        mat(r, c) = static_cast<float>(v[idx]);
    }
    }
    }
}
else if (mat_var.holds_type<uint64_t>())
{
    const auto& v = mat_var.get_vector<uint64_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < v.size())
            {
                mat(r, c) = static_cast<float>(v[idx]);
            }
        }
    }
}
else
{
    return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace utils
} // namespace matio
}
}
}
}
else if (mat_var.holds_type<uint64_t>())
{
    const auto& v = mat_var.get_vector<uint64_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < v.size())
            {
                mat(r, c) = static_cast<float>(v[idx]);
            }
        }
    }
}
else
{
    return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace utils
} // namespace matio
// MatFileUtils.cpp — single clean implementation
#include <cstdint>
#include <iostream>

#include "MatFileUtils.h"

namespace matio::utils
{

static auto build_matrix_from_matvar(const matio::MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    // Determine shape
    int rows = 1;
    int cols = 1;
    if (!mat_var.dimensions.empty())
    {
        rows = (mat_var.dimensions.size() >= 1) ? mat_var.dimensions[0] : 1;
        cols = (mat_var.dimensions.size() >= 2) ? mat_var.dimensions[1] : 1;
    }

    if (rows <= 0 || cols <= 0)
    {
        return std::nullopt;
    }

    Eigen::MatrixXf mat(rows, cols);
    mat.setZero();

    try
    {
        // row-major copy: idx = r * cols + c
        if (mat_var.holds_type<double>())
        {
            const auto& v = mat_var.get_vector<double>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<float>())
        {
            const auto& v = mat_var.get_vector<float>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = v[idx];
                    }
                }
            }
        }
        else if (mat_var.holds_type<int32_t>())
        {
            const auto& v = mat_var.get_vector<int32_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<int8_t>())
        {
            const auto& v = mat_var.get_vector<int8_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<uint8_t>())
        {
            const auto& v = mat_var.get_vector<uint8_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<int16_t>())
        {
            const auto& v = mat_var.get_vector<int16_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<uint16_t>())
        {
            const auto& v = mat_var.get_vector<uint16_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<uint32_t>())
        {
            const auto& v = mat_var.get_vector<uint32_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<int64_t>())
        {
            const auto& v = mat_var.get_vector<int64_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else if (mat_var.holds_type<uint64_t>())
        {
            const auto& v = mat_var.get_vector<uint64_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < v.size())
                    {
                        mat(r, c) = static_cast<float>(v[idx]);
                    }
                }
            }
        }
        else
        {
            return std::nullopt; // unsupported type
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
// MatFileUtils.cpp — single clean implementation
#include <cstdint>
#include <iostream>

#include "MatFileUtils.h"

    namespace matio::utils
    {

    static auto build_matrix_from_matvar(const matio::MatVar& mat_var)
        -> std::optional<Eigen::MatrixXf>
    {
        // Determine shape
        int rows = 1;
        int cols = 1;
        if (!mat_var.dimensions.empty())
        {
            rows = (mat_var.dimensions.size() >= 1) ? mat_var.dimensions[0] : 1;
            cols = (mat_var.dimensions.size() >= 2) ? mat_var.dimensions[1] : 1;
        }

        if (rows <= 0 || cols <= 0)
        {
            return std::nullopt;
        }

        Eigen::MatrixXf mat(rows, cols);
        mat.setZero();

        try
        {
            // row-major copy: idx = r * cols + c
            if (mat_var.holds_type<double>())
            {
                const auto& v = mat_var.get_vector<double>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<float>())
            {
                const auto& v = mat_var.get_vector<float>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = v[idx];
                        }
                    }
                }
            }
            else if (mat_var.holds_type<int32_t>())
            {
                const auto& v = mat_var.get_vector<int32_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<int8_t>())
            {
                const auto& v = mat_var.get_vector<int8_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<uint8_t>())
            {
                const auto& v = mat_var.get_vector<uint8_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<int16_t>())
            {
                const auto& v = mat_var.get_vector<int16_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<uint16_t>())
            {
                const auto& v = mat_var.get_vector<uint16_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<uint32_t>())
            {
                const auto& v = mat_var.get_vector<uint32_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<int64_t>())
            {
                const auto& v = mat_var.get_vector<int64_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else if (mat_var.holds_type<uint64_t>())
            {
                const auto& v = mat_var.get_vector<uint64_t>();
                for (int r = 0; r < rows; ++r)
                {
                    for (int c = 0; c < cols; ++c)
                    {
                        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                     static_cast<size_t>(c);
                        if (idx < v.size())
                        {
                            mat(r, c) = static_cast<float>(v[idx]);
                        }
                    }
                }
            }
            else
            {
                return std::nullopt; // unsupported type
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

        const matio::MatVar& var = var_opt.value();
        return build_matrix_from_matvar(var);
    }

    } // namespace matio::utils
    {
        mat(r, c) = static_cast<float>(vec[idx]);
    }
}
}
}
else if (mat_var.holds_type<int16_t>())
{
    const auto& vec = mat_var.get_vector<int16_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < vec.size())
            {
                mat(r, c) = static_cast<float>(vec[idx]);
            }
        }
    }
}
else if (mat_var.holds_type<uint16_t>())
{
    const auto& vec = mat_var.get_vector<uint16_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < vec.size())
            {
                mat(r, c) = static_cast<float>(vec[idx]);
            }
        }
    }
}
else if (mat_var.holds_type<uint32_t>())
{
    const auto& vec = mat_var.get_vector<uint32_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < vec.size())
            {
                mat(r, c) = static_cast<float>(vec[idx]);
            }
        }
    }
}
else if (mat_var.holds_type<int64_t>())
{
    const auto& vec = mat_var.get_vector<int64_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < vec.size())
            {
                mat(r, c) = static_cast<float>(vec[idx]);
            }
        }
    }
}
else if (mat_var.holds_type<uint64_t>())
{
    const auto& vec = mat_var.get_vector<uint64_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < vec.size())
            {
                mat(r, c) = static_cast<float>(vec[idx]);
            }
        }
    }
}
else
{
    return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils
{
    for (int c = 0; c < cols; ++c)
    {
        size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
        if (idx < vec.size())
        {
            mat(r, c) = static_cast<float>(vec[idx]);
        }
    }
}
}
else if (mat_var.holds_type<uint64_t>())
{
    const auto& vec = mat_var.get_vector<uint64_t>();
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            size_t idx =
                (static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(c);
            if (idx < vec.size())
            {
                mat(r, c) = static_cast<float>(vec[idx]);
            }
        }
    }
}
else
{
    return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils
// MatFileUtils.cpp — clean, single implementation
#include <algorithm>
#include <cstdint>
#include <iostream>

#include "MatFileUtils.h"

namespace matio::utils
{

static auto build_matrix_from_matvar(const MatVar& mat_var) -> std::optional<Eigen::MatrixXf>
{
    // Determine shape using MatVar dimensions (fallback 1x1)
    int rows = 1;
    int cols = 1;
    if (!mat_var.dimensions.empty())
    {
        rows = mat_var.dimensions.size() >= 1 ? mat_var.dimensions[0] : 1;
        cols = mat_var.dimensions.size() >= 2 ? mat_var.dimensions[1] : 1;
    }

    if (rows <= 0 || cols <= 0)
    {
        return std::nullopt;
    }

    Eigen::MatrixXf mat(rows, cols);
    mat.setZero();

    try
    {
        // copy using row-major indexing: idx = (r * cols) + c
        if (mat_var.holds_type<double>())
        {
            const auto& vec = mat_var.get_vector<double>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<float>())
        {
            const auto& vec = mat_var.get_vector<float>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = vec[idx];
                }
            }
        }
        else if (mat_var.holds_type<int32_t>())
        {
            const auto& vec = mat_var.get_vector<int32_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<int8_t>())
        {
            const auto& vec = mat_var.get_vector<int8_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<uint8_t>())
        {
            const auto& vec = mat_var.get_vector<uint8_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<int16_t>())
        {
            const auto& vec = mat_var.get_vector<int16_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<uint16_t>())
        {
            const auto& vec = mat_var.get_vector<uint16_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<uint32_t>())
        {
            const auto& vec = mat_var.get_vector<uint32_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<int64_t>())
        {
            const auto& vec = mat_var.get_vector<int64_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else if (mat_var.holds_type<uint64_t>())
        {
            const auto& vec = mat_var.get_vector<uint64_t>();
            for (int r = 0; r < rows; ++r)
            {
                for (int c = 0; c < cols; ++c)
                {
                    size_t idx = (static_cast<size_t>(r) * static_cast<size_t>(cols)) +
                                 static_cast<size_t>(c);
                    if (idx < vec.size()) mat(r, c) = static_cast<float>(vec[idx]);
                }
            }
        }
        else
        {
            return std::nullopt; // unsupported type
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

    const matio::MatVar& var = var_opt.value();
    return build_matrix_from_matvar(var);
}

} // namespace matio::utils
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

const MatVar& var = var_opt.value();
return build_matrix_from_matvar(var);
}

} // namespace matio::utils
