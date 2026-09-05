/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendDetail.hpp
 * @brief Host storage and OpenCL helper primitives shared by every
 *        OpenCLTensorBackend*.cpp translation unit.
 *
 * Everything below is `inline` and lives directly in `namespace nn` (not a
 * nested namespace) so that it is both a single shared definition across
 * translation units (the ODR guarantee for inline functions) and callable
 * unqualified exactly as it was when these were file-local anonymous-
 * namespace helpers in one big OpenCLTensorBackend.cpp. This matters for
 * warn_opencl_cpu_fallback_once specifically: its "already warned" dedup set
 * is a function-local static, and the inline-function ODR rule is what keeps
 * that set a single instance program-wide instead of one per translation
 * unit that includes this header.
 */
#pragma once

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "logging/Logger.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace nn
{

class OpenCLHostStorage
{
   public:
    OpenCLHostStorage() = default;

    explicit OpenCLHostStorage(Index rows, Index cols)
        : m_shape({rows, cols}), m_data(rows * cols, 0.0F)
    {
    }

    explicit OpenCLHostStorage(Index d1, Index d2, Index d3)
        : m_shape({d1, d2, d3}), m_data(d1 * d2 * d3, 0.0F)
    {
    }

    explicit OpenCLHostStorage(Index d1, Index d2, Index d3, Index d4)
        : m_shape({d1, d2, d3, d4}), m_data(d1 * d2 * d3 * d4, 0.0F)
    {
    }

    explicit OpenCLHostStorage(const std::vector<Index>& shape)
        : m_shape(shape), m_data(total_size(shape), 0.0F)
    {
    }

    const std::vector<Index>& shape() const
    {
        return m_shape;
    }

    void reshape(const std::vector<Index>& new_shape)
    {
        if (total_size(new_shape) != size())
        {
            throw std::invalid_argument("Reshape total size mismatch");
        }
        // Reshape must reinterpret the tensor in ROW-MAJOR logical order to
        // match the XTensor backend (backend-parity contract). This storage is
        // column-major, so a metadata-only shape swap would silently produce a
        // different logical tensor. Permute the buffer so element k of the
        // row-major traversal is preserved across the reshape. 1-D storage is
        // layout-free; the permutation only matters when either side is >= 2-D.
        if (m_shape.size() >= 2 || new_shape.size() >= 2)
        {
            std::vector<float> permuted(m_data.size());
            const Index n = size();
            for (Index k = 0; k < n; ++k)
            {
                permuted[row_major_to_storage(new_shape, k)] =
                    m_data[row_major_to_storage(m_shape, k)];
            }
            m_data = std::move(permuted);
        }
        m_shape = new_shape;
    }

    Index rows() const
    {
        return m_shape.empty() ? 0 : m_shape[0];
    }

    Index cols() const
    {
        return m_shape.size() < 2 ? 1 : m_shape[1];
    }

    Index size() const
    {
        return m_data.size();
    }

    float* mutable_data_ptr()
    {
        return m_data.data();
    }

    const float* data_ptr() const
    {
        return m_data.data();
    }

    void fill(float value)
    {
        std::fill(m_data.begin(), m_data.end(), value);
    }

    float& at(Index i)
    {
        if (i >= size())
        {
            throw std::out_of_range("Index out of range");
        }
        return m_data[i];
    }

    const float& at(Index i) const
    {
        if (i >= size())
        {
            throw std::out_of_range("Index out of range");
        }
        return m_data[i];
    }

    float& at(Index row, Index col)
    {
        return m_data[offset_2d(row, col)];
    }

    const float& at(Index row, Index col) const
    {
        return m_data[offset_2d(row, col)];
    }

    float& at(Index d1, Index d2, Index d3)
    {
        return m_data[offset_3d(d1, d2, d3)];
    }

    const float& at(Index d1, Index d2, Index d3) const
    {
        return m_data[offset_3d(d1, d2, d3)];
    }

    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        return m_data[offset_4d(d1, d2, d3, d4)];
    }

    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        return m_data[offset_4d(d1, d2, d3, d4)];
    }

    float& at(const std::vector<Index>& indices)
    {
        return m_data[offset_nd(indices)];
    }

    const float& at(const std::vector<Index>& indices) const
    {
        return m_data[offset_nd(indices)];
    }

   private:
    /// Row-major linear index k (XTensor logical order, last dim fastest) →
    /// this storage's column-major offset (first dim fastest).
    static Index row_major_to_storage(const std::vector<Index>& shape, Index k)
    {
        if (shape.size() <= 1)
        {
            return k;
        }
        std::vector<Index> idx(shape.size());
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            idx[static_cast<std::size_t>(d)] = k % shape[static_cast<std::size_t>(d)];
            k /= shape[static_cast<std::size_t>(d)];
        }
        Index off = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            off = idx[static_cast<std::size_t>(d)] + shape[static_cast<std::size_t>(d)] * off;
        }
        return off;
    }

    static Index total_size(const std::vector<Index>& shape)
    {
        return std::accumulate(shape.begin(),
            shape.end(),
            static_cast<Index>(1),
            [](Index acc, Index dim) { return acc * dim; });
    }

    Index offset_2d(Index row, Index col) const
    {
        if (m_shape.size() != 2)
        {
            throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
        }
        if (row >= m_shape[0] || col >= m_shape[1])
        {
            throw std::out_of_range("Index out of range");
        }
        return row + (col * m_shape[0]);
    }

    Index offset_3d(Index d1, Index d2, Index d3) const
    {
        if (m_shape.size() != 3)
        {
            throw std::invalid_argument("at(d1, d2, d3) is only valid for 3D tensors");
        }
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2])
        {
            throw std::out_of_range("Index out of range");
        }
        const Index col_idx = d2 + d3 * m_shape[1];
        return d1 + (col_idx * m_shape[0]);
    }

    Index offset_4d(Index d1, Index d2, Index d3, Index d4) const
    {
        if (m_shape.size() != 4)
        {
            throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
        }
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
        {
            throw std::out_of_range("Index out of range");
        }

        const Index col_idx = d2 + (d3 + d4 * m_shape[2]) * m_shape[1];
        return d1 + (col_idx * m_shape[0]);
    }

    Index offset_nd(const std::vector<Index>& indices) const
    {
        if (indices.size() != m_shape.size())
        {
            throw std::invalid_argument("Indices dimension mismatch");
        }

        if (indices.size() == 1)
        {
            return indices[0];
        }
        if (indices.size() == 2)
        {
            return offset_2d(indices[0], indices[1]);
        }
        if (indices.size() == 3)
        {
            return offset_3d(indices[0], indices[1], indices[2]);
        }
        if (indices.size() == 4)
        {
            return offset_4d(indices[0], indices[1], indices[2], indices[3]);
        }

        Index flat_idx = 0;
        Index stride = 1;
        for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
        {
            if (indices[static_cast<std::size_t>(i)] >= m_shape[static_cast<std::size_t>(i)])
            {
                throw std::out_of_range("Index out of range");
            }
            flat_idx += indices[static_cast<std::size_t>(i)] * stride;
            stride *= m_shape[static_cast<std::size_t>(i)];
        }
        return flat_idx;
    }

    std::vector<Index> m_shape;
    std::vector<float> m_data;
};

inline void check_cl_error(cl_int err, const char* context)
{
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error(
            std::string("OpenCL error in ") + context + ": " + std::to_string(err));
    }
}

inline void finish_queue_if_not_batching(cl_command_queue queue, const char* context)
{
    if (opencl::OpenCLContext::is_batching())
    {
        return;
    }
    check_cl_error(clFinish(queue), context);
}

inline bool can_use_opencl()
{
    // OpenCL runtime is intentionally disabled under ASan to avoid false positives
    // from third-party drivers while preserving CPU fallback semantics.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        return false;
    }
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && ((__SANITIZE_ADDRESS__ + 0) == 1)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        return false;
    }
#endif
    volatile bool runtime_available = opencl::OpenCLContext::instance().is_available();
    return runtime_available;
}

inline void warn_opencl_cpu_fallback_once(const std::string& operation, const std::string& reason)
{
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned_messages;

    const std::string message = "OPENCL BACKEND CANNOT EXECUTE " + operation + ": " + reason;
    std::lock_guard<std::mutex> lock(warned_mutex);
    if (warned_messages.insert(message).second)
    {
        NN_LOG_WARN(message);
    }
}

inline bool can_use_opencl(const char* operation)
{
#if defined(__SANITIZE_ADDRESS__) && ((__SANITIZE_ADDRESS__ + 0) == 1)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        warn_opencl_cpu_fallback_once(
            operation, "AddressSanitizer build disables OpenCL execution");
        return false;
    }
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        warn_opencl_cpu_fallback_once(
            operation, "AddressSanitizer build disables OpenCL execution");
        return false;
    }
#endif
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
#else
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
#endif
}

[[noreturn]] inline void throw_opencl_only_failure(
    const std::string& operation, const std::string& reason)
{
    throw std::runtime_error("OpenCL-only backend failure in " + operation + ": " + reason);
}

inline std::size_t round_up(std::size_t global, std::size_t local)
{
    if (local == 0)
    {
        return global;
    }
    const std::size_t rem = global % local;
    return rem == 0 ? global : (global + (local - rem));
}

inline void copy_host_to_device(cl_command_queue queue,
    cl_mem device_buffer,
    const float* host_data,
    std::size_t bytes,
    const char* context)
{
    check_cl_error(clEnqueueWriteBuffer(
                       queue, device_buffer, CL_TRUE, 0, bytes, host_data, 0, nullptr, nullptr),
        context);
}

inline void copy_host_to_device_async(cl_command_queue queue,
    cl_mem device_buffer,
    const float* host_data,
    std::size_t bytes,
    const char* context,
    cl_event* out_event = nullptr)
{
    if (bytes > 0)
    {
        check_cl_error(
            clEnqueueWriteBuffer(
                queue, device_buffer, CL_FALSE, 0, bytes, host_data, 0, nullptr, out_event),
            context);
    }
}

inline void copy_device_to_host(cl_command_queue queue,
    cl_mem device_buffer,
    float* host_data,
    std::size_t bytes,
    const char* context)
{
    check_cl_error(clEnqueueReadBuffer(
                       queue, device_buffer, CL_TRUE, 0, bytes, host_data, 0, nullptr, nullptr),
        context);
}

inline void copy_device_to_host_async(cl_command_queue queue,
    cl_mem device_buffer,
    float* host_data,
    std::size_t bytes,
    const char* context,
    cl_event* out_event = nullptr)
{
    if (bytes > 0)
    {
        check_cl_error(
            clEnqueueReadBuffer(
                queue, device_buffer, CL_FALSE, 0, bytes, host_data, 0, nullptr, out_event),
            context);
    }
}

} // namespace nn
