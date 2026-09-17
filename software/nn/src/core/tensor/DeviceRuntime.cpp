/**
 * @file src/core/tensor/DeviceRuntime.cpp
 * @brief Implementation of DeviceRuntime. CPU is the only Device left after
 *        the OpenCL/SYCL backends (and their runtime-scope setup) were
 *        removed, so there is nothing to lazily initialize here anymore.
 */

#include "device/DeviceRuntime.hpp"

namespace nn
{

void DeviceRuntime::ensure_runtime(const Device& /*device*/)
{
    // no-op: CPU needs no runtime warm-up.
}

} // namespace nn
