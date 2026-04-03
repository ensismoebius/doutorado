/**
 * @file src/core/tensor/OpenCLTensorBackend.cpp
 * @brief OpenCL tensor backend implementation (Phase 1: CPU fallback).
 *
 * Phase 1 delegates all operations to EigenTensorBackend for correctness.
 * GPU implementations will be added incrementally as needed.
 */

#include "nn/tensor/OpenCLTensorBackend.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/DeviceMemory.hpp"
#include "nn/tensor/EigenTensorBackend.hpp"
#include "nn/tensor/KernelManager.hpp"
#include "nn/tensor/OpenCLContext.hpp"

namespace nn
{

namespace
{
void check_cl_error(cl_int err, const char* context)
{
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error(
            std::string("OpenCL error in ") + context + ": " + std::to_string(err));
    }
}

bool can_use_opencl()
{
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    return false;
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    return false;
#endif
    return opencl::OpenCLContext::instance().is_available();
}

std::size_t round_up(std::size_t global, std::size_t local)
{
    if (local == 0)
    {
        return global;
    }
    const std::size_t rem = global % local;
    return rem == 0 ? global : (global + (local - rem));
}
} // namespace

// Constructors
OpenCLTensorBackend::OpenCLTensorBackend(Index rows, Index cols)
    : m_backend(std::make_unique<EigenTensorBackend>(rows, cols))
{
}

OpenCLTensorBackend::OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4)
    : m_backend(std::make_unique<EigenTensorBackend>(d1, d2, d3, d4))
{
}

OpenCLTensorBackend::OpenCLTensorBackend(const std::vector<Index>& shape)
    : m_backend(std::make_unique<EigenTensorBackend>(shape))
{
}

OpenCLTensorBackend::OpenCLTensorBackend(const OpenCLTensorBackend& other)
{
    if (other.m_backend)
    {
        m_backend = std::make_unique<EigenTensorBackend>(*other.m_backend);
    }
    if (other.m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend);
    }
}

OpenCLTensorBackend& OpenCLTensorBackend::operator=(const OpenCLTensorBackend& other)
{
    if (this != &other)
    {
        m_backend =
            other.m_backend ? std::make_unique<EigenTensorBackend>(*other.m_backend) : nullptr;
        m_grad_backend = other.m_grad_backend
                             ? std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend)
                             : nullptr;
    }
    return *this;
}

OpenCLTensorBackend::~OpenCLTensorBackend() = default;

// Static Factories
OpenCLTensorBackend OpenCLTensorBackend::zeros(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::zeros(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::ones(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::ones(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::random(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols, std::mt19937& rng)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::random(rows, cols, rng);
    return t;
}

// Shape & Access
const std::vector<Index>& OpenCLTensorBackend::shape() const
{
    return m_backend->shape();
}

void OpenCLTensorBackend::reshape(const std::vector<Index>& new_shape)
{
    m_backend->reshape(new_shape);
}

Index OpenCLTensorBackend::rows() const
{
    return m_backend->rows();
}

Index OpenCLTensorBackend::cols() const
{
    return m_backend->cols();
}

Index OpenCLTensorBackend::size() const
{
    return m_backend->size();
}

// N-D access
float& OpenCLTensorBackend::at(Index i)
{
    return m_backend->at(i);
}

const float& OpenCLTensorBackend::at(Index i) const
{
    return m_backend->at(i);
}

float& OpenCLTensorBackend::at(Index row, Index col)
{
    return m_backend->at(row, col);
}

const float& OpenCLTensorBackend::at(Index row, Index col) const
{
    return m_backend->at(row, col);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    return m_backend->at(d1, d2, d3, d4);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    return m_backend->at(d1, d2, d3, d4);
}

float& OpenCLTensorBackend::at(const std::vector<Index>& indices)
{
    return m_backend->at(indices);
}

const float& OpenCLTensorBackend::at(const std::vector<Index>& indices) const
{
    return m_backend->at(indices);
}

// In-place operations
void OpenCLTensorBackend::add_inplace(const OpenCLTensorBackend& other)
{
    m_backend->add_inplace(*other.m_backend);
}

void OpenCLTensorBackend::subtract_inplace(const OpenCLTensorBackend& other)
{
    m_backend->subtract_inplace(*other.m_backend);
}

void OpenCLTensorBackend::multiply_inplace(const OpenCLTensorBackend& other)
{
    m_backend->multiply_inplace(*other.m_backend);
}

void OpenCLTensorBackend::divide_inplace(const OpenCLTensorBackend& other)
{
    m_backend->divide_inplace(*other.m_backend);
}

void OpenCLTensorBackend::add_scalar_inplace(float val)
{
    m_backend->add_scalar_inplace(val);
}

void OpenCLTensorBackend::multiply_scalar_inplace(float val)
{
    m_backend->multiply_scalar_inplace(val);
}

void OpenCLTensorBackend::divide_scalar_inplace(float val)
{
    m_backend->divide_scalar_inplace(val);
}

void OpenCLTensorBackend::sqrt_inplace()
{
    m_backend->sqrt_inplace();
}

void OpenCLTensorBackend::square_inplace()
{
    m_backend->square_inplace();
}

void OpenCLTensorBackend::add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector)
{
    m_backend->add_col_vector_to_rows_inplace(*col_vector.m_backend);
}

// Element-wise operations
OpenCLTensorBackend OpenCLTensorBackend::exp() const
{
    if (can_use_opencl())
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            EigenTensorBackend out(shape());
            opencl::DeviceMemory input_dev(n * sizeof(float));
            opencl::DeviceMemory out_dev(n * sizeof(float));

            input_dev.copy_to_device(m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("exp_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(exp, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(exp, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(exp, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(exp)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(exp)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL exp fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->exp());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::sqrt() const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->sqrt());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::square() const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->square());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::add(const OpenCLTensorBackend& other) const
{
    if (can_use_opencl() && shape() == other.shape())
    {
        try
        {
            const auto n = size();
            EigenTensorBackend out(shape());
            opencl::DeviceMemory a_dev(n * sizeof(float));
            opencl::DeviceMemory b_dev(n * sizeof(float));
            opencl::DeviceMemory out_dev(n * sizeof(float));

            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("add_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(add, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(add, b)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "clSetKernelArg(add, out)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clSetKernelArg(add, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(add)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(add)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->add(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::subtract(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->subtract(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::multiply(const OpenCLTensorBackend& other) const
{
    if (can_use_opencl() && shape() == other.shape())
    {
        try
        {
            const auto n = size();
            EigenTensorBackend out(shape());
            opencl::DeviceMemory a_dev(n * sizeof(float));
            opencl::DeviceMemory b_dev(n * sizeof(float));
            opencl::DeviceMemory out_dev(n * sizeof(float));

            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("multiply_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(multiply, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(multiply, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(multiply, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(multiply, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(multiply)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->multiply(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::divide(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->divide(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::add_scalar(float val) const
{
    if (can_use_opencl())
    {
        try
        {
            const auto n = size();
            EigenTensorBackend out(shape());
            opencl::DeviceMemory input_dev(n * sizeof(float));
            opencl::DeviceMemory out_dev(n * sizeof(float));

            input_dev.copy_to_device(m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("add_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(add_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(add_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                "clSetKernelArg(add_scalar, scalar)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(add_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(add_scalar)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(add_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->add_scalar(val));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::multiply_scalar(float val) const
{
    if (can_use_opencl())
    {
        try
        {
            const auto n = size();
            EigenTensorBackend out(shape());
            opencl::DeviceMemory input_dev(n * sizeof(float));
            opencl::DeviceMemory out_dev(n * sizeof(float));

            input_dev.copy_to_device(m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("multiply_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(multiply_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(multiply_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                "clSetKernelArg(multiply_scalar, scalar)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(multiply_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(multiply_scalar)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->multiply_scalar(val));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::divide_scalar(float val) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->divide_scalar(val));
    return t;
}

// Reduction
OpenCLTensorBackend OpenCLTensorBackend::rowwise_sum() const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->rowwise_sum());
    return t;
}

// Linear algebra
OpenCLTensorBackend OpenCLTensorBackend::matmul(const OpenCLTensorBackend& other) const
{
    if (can_use_opencl() && shape().size() == 2 && other.shape().size() == 2 &&
        cols() == other.rows())
    {
        try
        {
            const Index m = rows();
            const Index k = cols();
            const Index n = other.cols();

            EigenTensorBackend out(m, n);
            opencl::DeviceMemory a_dev(m * k * sizeof(float));
            opencl::DeviceMemory b_dev(k * n * sizeof(float));
            opencl::DeviceMemory out_dev(m * n * sizeof(float));

            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("matmul_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem c_mem = out_dev.get_device_buffer();
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(matmul, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(matmul, b)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem), "clSetKernelArg(matmul, c)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32), "clSetKernelArg(matmul, m)");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "clSetKernelArg(matmul, n)");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32), "clSetKernelArg(matmul, k)");

            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(matmul)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL matmul fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->matmul(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend =
        std::make_unique<EigenTensorBackend>(m_backend->matmul_transposed(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::transpose() const
{
    if (can_use_opencl() && shape().size() == 2)
    {
        try
        {
            const Index in_rows = rows();
            const Index in_cols = cols();

            EigenTensorBackend out(in_cols, in_rows);
            opencl::DeviceMemory in_dev(in_rows * in_cols * sizeof(float));
            opencl::DeviceMemory out_dev(in_rows * in_cols * sizeof(float));

            in_dev.copy_to_device(m_backend->data_ptr());

            auto& ctx = opencl::OpenCLContext::instance();
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("transpose_kernel");
            const cl_mem in_mem = in_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint rows_u32 = static_cast<cl_uint>(in_rows);
            const cl_uint cols_u32 = static_cast<cl_uint>(in_cols);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(transpose, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(transpose, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                "clSetKernelArg(transpose, rows)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                "clSetKernelArg(transpose, cols)");

            const std::size_t global[2] = {in_rows, in_cols};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(transpose)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(transpose)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL transpose fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->transpose());
    return t;
}

// Comparisons
OpenCLTensorBackend OpenCLTensorBackend::compare_lt(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_lt(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_gt(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_le(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_ge(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq(const OpenCLTensorBackend& other) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_eq(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_lt_scalar(float value) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_lt_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt_scalar(float value) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_gt_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le_scalar(float value) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_le_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge_scalar(float value) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_ge_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq_scalar(float value) const
{
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_eq_scalar(value));
    return t;
}

// Gradient management
OpenCLTensorBackend& OpenCLTensorBackend::grad_ref()
{
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(shape());
        m_grad_backend->m_backend = std::make_unique<EigenTensorBackend>(shape());
    }
    return *m_grad_backend;
}

const OpenCLTensorBackend& OpenCLTensorBackend::get_grad() const
{
    if (!m_grad_backend)
    {
        throw std::runtime_error("Gradient not allocated");
    }
    return *m_grad_backend;
}

void OpenCLTensorBackend::zero_grad()
{
    if (m_grad_backend)
    {
        *m_grad_backend->m_backend = EigenTensorBackend::zeros(rows(), cols());
    }
}

} // namespace nn
