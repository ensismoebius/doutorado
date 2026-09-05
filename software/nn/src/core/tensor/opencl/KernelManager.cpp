/**
 * @file src/core/tensor/opencl/KernelManager.cpp
 * @brief OpenCL kernel compilation and caching implementation.
 */

#include "tensor/opencl/KernelManager.hpp"

#include <cassert>
#include <stdexcept>

#include "KernelSourceElementWise.hpp"
#include "KernelSourceFused.hpp"
#include "KernelSourceLinearAlgebra.hpp"
#include "KernelSourceReductions.hpp"
#include "logging/Logger.hpp"
#include "tensor/opencl/OpenCLContext.hpp"

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

std::string KernelManager::get_kernel_source(const std::string& program_name)
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
    if (kernel_name == "matmul_rhs_transposed_bias_relu_kernel" ||
        kernel_name == "matmul_rhs_transposed_bias_leaky_relu_kernel" ||
        kernel_name == "matmul_rhs_transposed_bias_sigmoid_kernel" ||
        kernel_name == "matmul_rhs_transposed_bias_tanh_kernel")
    {
        program_name = "fused";
    }
    else if (kernel_name == "strided_copy_2d_kernel" ||
             kernel_name.find("matmul") != std::string::npos ||
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
             kernel_name.find("compare") != std::string::npos ||
             kernel_name.find("fill") != std::string::npos ||
             kernel_name.find("abs") != std::string::npos ||
             kernel_name.find("clamp") != std::string::npos)
    {
        program_name = "element_wise";
    }
    else if (kernel_name.find("rowwise_sum") != std::string::npos ||
             kernel_name.find("sum") != std::string::npos)
    {
        program_name = "reductions";
    }
    else if (kernel_name.find("mul_add") != std::string::npos ||
             kernel_name.find("relu") != std::string::npos ||
             kernel_name.find("mse") != std::string::npos ||
             kernel_name.find("lif") != std::string::npos ||
             kernel_name.find("adam") != std::string::npos)
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
