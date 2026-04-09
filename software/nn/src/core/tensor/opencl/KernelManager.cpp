/**
 * @file src/core/tensor/opencl/KernelManager.cpp
 * @brief OpenCL kernel compilation and caching implementation.
 */

#include "nn/tensor/opencl/KernelManager.hpp"

#include <cassert>
#include <stdexcept>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/opencl/OpenCLContext.hpp"

namespace nn::opencl
{

// Helper: throw on OpenCL error
static void check_cl_error(cl_int err, const std::string& context)
{
    if (err != CL_SUCCESS)
    {
        std::string msg = "OpenCL error [" + context + "]: " + std::to_string(err);
        throw std::runtime_error(msg);
    }
}

// Embedded kernel sources
static constexpr const char* KERNEL_SOURCE_LINEAR_ALGEBRA = R"(
__kernel void matmul_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        // Eigen::MatrixXf is column-major by default:
        // idx(row,col) = row + col * rows
        sum += A[row + i * M] * B[i + col * K];
    }
    
    C[row + col * M] = sum;
}

__kernel void transpose_kernel(
    __global const float* input,
    __global float* output,
    const uint rows,
    const uint cols
) {
    const uint row = get_global_id(0);
    const uint col = get_global_id(1);
    
    if (row >= rows || col >= cols) return;
    
    // Column-major transpose mapping:
    // input(row,col)  idx = row + col * rows
    // output(col,row) idx = col + row * cols
    output[col + row * cols] = input[row + col * rows];
}
)";

static constexpr const char* KERNEL_SOURCE_ELEMENT_WISE = R"(
__kernel void add_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] + B[idx];
}

__kernel void multiply_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] * B[idx];
}

__kernel void subtract_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] - B[idx];
}

__kernel void divide_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] / (B[idx] + 1e-8f);
}

__kernel void exp_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = exp(input[idx]);
}

__kernel void sqrt_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = sqrt(max(input[idx], 0.0f));
}

__kernel void square_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] * input[idx];
}

__kernel void add_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] + scalar;
}

__kernel void multiply_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] * scalar;
}

__kernel void divide_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] / (scalar + 1e-8f);
}

__kernel void add_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] + B[idx];
}

__kernel void subtract_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] - B[idx];
}

__kernel void multiply_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] * B[idx];
}

__kernel void divide_inplace_kernel(
    __global float* A,
    __global const float* B,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    A[idx] = A[idx] / (B[idx] + 1e-8f);
}

__kernel void add_scalar_inplace_kernel(
    __global float* data,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] + scalar;
}

__kernel void multiply_scalar_inplace_kernel(
    __global float* data,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] * scalar;
}

__kernel void divide_scalar_inplace_kernel(
    __global float* data,
    const float scalar,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] / (scalar + 1e-8f);
}

__kernel void sqrt_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = sqrt(max(data[idx], 0.0f));
}

__kernel void square_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = data[idx] * data[idx];
}

__kernel void add_col_vector_to_rows_kernel(
    __global float* data,
    __global const float* col_vector,
    const uint rows,
    const uint cols
) {
    const uint idx = get_global_id(0);
    if (idx >= rows * cols) return;
    const uint col = idx / rows;
    data[idx] = data[idx] + col_vector[col];
}

__kernel void compare_lt_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] < B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_gt_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] > B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_le_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] <= B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_ge_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = A[idx] >= B[idx] ? 1.0f : 0.0f;
}

__kernel void compare_eq_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    C[idx] = fabs(A[idx] - B[idx]) < 1e-6f ? 1.0f : 0.0f;
}

__kernel void compare_lt_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] < value ? 1.0f : 0.0f;
}

__kernel void compare_gt_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] > value ? 1.0f : 0.0f;
}

__kernel void compare_le_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] <= value ? 1.0f : 0.0f;
}

__kernel void compare_ge_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = input[idx] >= value ? 1.0f : 0.0f;
}

__kernel void compare_eq_scalar_kernel(
    __global const float* input,
    __global float* output,
    const float value,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = fabs(input[idx] - value) < 1e-6f ? 1.0f : 0.0f;
}
)";

static constexpr const char* KERNEL_SOURCE_REDUCTIONS = R"(
__kernel void rowwise_sum_kernel(
    __global const float* input,
    __global float* output,
    const uint rows,
    const uint cols
) {
    const uint row = get_global_id(0);

    if (row >= rows) return;

    float sum = 0.0f;
    for (uint col = 0; col < cols; ++col) {
        sum += input[row + col * rows];
    }

    output[row] = sum;
}
)";

static constexpr const char* KERNEL_SOURCE_FUSED = R"(
__kernel void mul_add_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* c,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = a[idx] * b[idx] + c[idx];
}

__kernel void mul_add_sigmoid_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* c,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float val = a[idx] * b[idx] + c[idx];
    output[idx] = 1.0f / (1.0f + exp(-val));
}

__kernel void relu_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = fmax(input[idx], 0.0f);
}

__kernel void relu_inplace_kernel(
    __global float* data,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    data[idx] = fmax(data[idx], 0.0f);
}

__kernel void add_bias_kernel(
    __global const float* input,
    __global const float* bias,
    __global float* output,
    const uint rows,
    const uint cols
) {
    const uint idx = get_global_id(0);
    if (idx >= rows * cols) return;
    const uint row = idx % rows;
    output[idx] = input[idx] + bias[row];
}

__kernel void matmul_add_bias_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* bias,
    __global float* output,
    const uint M,
    const uint N,
    const uint K
) {
    const uint row = get_global_id(0) / N;
    const uint col = get_global_id(0) % N;
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < K; ++i) {
        sum += a[row * K + i] * b[i * N + col];
    }
    output[row * N + col] = sum + bias[col];
}

__kernel void gelu_kernel(
    __global const float* input,
    __global float* output,
    const uint size
) {
    const uint idx = get_global_id(0);
    if (idx >= size) return;
    float x = input[idx];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}
)";

KernelManager& KernelManager::instance()
{
    static KernelManager mgr;
    return mgr;
}

KernelManager::KernelManager()
{
    NN_LOG_INFO("KernelManager initialized");
}

KernelManager::~KernelManager()
{
    release_all();
}

std::string KernelManager::get_kernel_source(const std::string& program_name) const
{
    if (program_name == "linear_algebra")
    {
        return KERNEL_SOURCE_LINEAR_ALGEBRA;
    }
    else if (program_name == "element_wise")
    {
        return KERNEL_SOURCE_ELEMENT_WISE;
    }
    else if (program_name == "reductions")
    {
        return KERNEL_SOURCE_REDUCTIONS;
    }
    else if (program_name == "fused")
    {
        return KERNEL_SOURCE_FUSED;
    }
    else
    {
        throw std::runtime_error("KernelManager: unknown program '" + program_name + "'");
    }
}

cl_program KernelManager::get_program(const std::string& program_name)
{
    // Check cache
    auto it = m_programs.find(program_name);
    if (it != m_programs.end())
    {
        NN_LOG_DEBUG("KernelManager: using cached program '" + program_name + "'");
        return it->second;
    }

    // Compile new program
    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        throw std::runtime_error("KernelManager: OpenCL context not available");
    }

    std::string source = get_kernel_source(program_name);
    const char* src = source.c_str();
    size_t src_len = source.length();

    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ctx.get_context(), 1, &src, &src_len, &err);
    check_cl_error(err, "clCreateProgramWithSource");

    // Build program
    cl_device_id device = ctx.get_device();
    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);

    if (err != CL_SUCCESS)
    {
        // Print build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);

        NN_LOG_ERROR("Kernel compilation failed for '" + program_name + "':\n" + log);

        clReleaseProgram(program);
        check_cl_error(err, "clBuildProgram");
    }

    NN_LOG_INFO("KernelManager: compiled program '" + program_name + "'");
    m_programs[program_name] = program;

    return program;
}

cl_kernel KernelManager::get_kernel(const std::string& kernel_name)
{
    // Check cache
    auto it = m_kernels.find(kernel_name);
    if (it != m_kernels.end())
    {
        NN_LOG_DEBUG("KernelManager: using cached kernel '" + kernel_name + "'");
        return it->second;
    }

    // Determine which program this kernel belongs to
    std::string program_name;
    if (kernel_name.find("matmul") != std::string::npos ||
        kernel_name.find("transpose") != std::string::npos)
    {
        program_name = "linear_algebra";
    }
    else if (kernel_name.find("add") != std::string::npos ||
             kernel_name.find("subtract") != std::string::npos ||
             kernel_name.find("multiply") != std::string::npos ||
             kernel_name.find("divide") != std::string::npos ||
             kernel_name.find("exp") != std::string::npos ||
             kernel_name.find("sqrt") != std::string::npos ||
             kernel_name.find("square") != std::string::npos ||
             kernel_name.find("compare") != std::string::npos)
    {
        program_name = "element_wise";
    }
    else if (kernel_name.find("rowwise_sum") != std::string::npos)
    {
        program_name = "reductions";
    }
    else if (kernel_name.find("mul_add") != std::string::npos)
    {
        program_name = "fused";
    }
    else
    {
        throw std::runtime_error("KernelManager: unknown kernel '" + kernel_name + "'");
    }

    // Get or compile program
    cl_program program = get_program(program_name);

    // Extract kernel from program
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    check_cl_error(err, "clCreateKernel '" + kernel_name + "'");

    NN_LOG_INFO("KernelManager: created kernel '" + kernel_name + "'");
    m_kernels[kernel_name] = kernel;

    return kernel;
}

bool KernelManager::has_kernel(const std::string& kernel_name) const
{
    return m_kernels.find(kernel_name) != m_kernels.end();
}

void KernelManager::release_all()
{
    for (auto& [name, kernel] : m_kernels)
    {
        clReleaseKernel(kernel);
    }
    m_kernels.clear();

    for (auto& [name, program] : m_programs)
    {
        clReleaseProgram(program);
    }
    m_programs.clear();

    NN_LOG_INFO("KernelManager: released all kernels and programs");
}

} // namespace nn::opencl
