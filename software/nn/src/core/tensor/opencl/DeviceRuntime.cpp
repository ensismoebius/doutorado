/**
 * @file src/core/tensor/opencl/DeviceRuntime.cpp
 * @brief Implementation of DeviceRuntime for OpenCL backend.
 */

#include "nn/device/Device.hpp"

namespace nn
{

void DeviceRuntime::ensure_runtime(const Device& device)
{
    if (!device.is_opencl()) return;
    static std::once_flag s_flag;
    static std::optional<nn::OpenCLTensorBackend::RuntimeScope> s_scope;
    std::call_once(s_flag,
        [&device]
        {
            s_scope =
                nn::OpenCLTensorBackend::start_runtime_scope_or_throw(device.profiling_enabled);
        });
}

} // namespace nn