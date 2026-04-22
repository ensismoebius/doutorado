#pragma once

#include <string>

#include "nn/device/DeviceType.hpp"

namespace nn
{

struct Device
{
    DeviceType type = DeviceType::CPU;
    std::string id = "cpu";
    bool profiling_enabled = false;

    static auto from_string(const std::string& s) -> Device
    {
        if (s.empty()) return Device{DeviceType::CPU, "cpu"};
        if (s == "cpu") return Device{DeviceType::CPU, s};
        if (s.rfind("opencl", 0) == 0) return Device{DeviceType::OPENCL, s};
        return Device{DeviceType::CPU, s};
    }

    auto with_profiling(bool enabled) const -> Device
    {
        Device d = *this;
        d.profiling_enabled = enabled;
        return d;
    }

    bool is_cpu() const
    {
        return type == DeviceType::CPU;
    }
    bool is_opencl() const
    {
        return type == DeviceType::OPENCL;
    }
    const std::string& to_string() const
    {
        return id;
    }
};

} // namespace nn