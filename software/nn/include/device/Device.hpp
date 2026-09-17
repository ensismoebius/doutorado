#pragma once

#include <string>

#include "device/DeviceType.hpp"

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
    const std::string& to_string() const
    {
        return id;
    }
};

} // namespace nn