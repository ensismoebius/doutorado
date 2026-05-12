#pragma once

#include "device/Device.hpp"

namespace nn
{

struct DeviceRuntime
{
    static void ensure_runtime(const Device& device);
};

} // namespace nn