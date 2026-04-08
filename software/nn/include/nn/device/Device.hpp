/**
 * @file include/nn/device/Device.hpp
 * @brief Lightweight Device abstraction with lazy process-level runtime management (PyTorch-like).
 *
 * Usage example (PyTorch-like):
 *   auto device = nn::Device::from_string(config.device)  // "cpu" or "opencl"
 *                     .with_profiling(config.opencl_profiling_enabled);
 *   model->to(device); // lazily starts the OpenCL runtime on first call; no-op for cpu
 *
 * The runtime is kept alive for the lifetime of the process through an internal static.
 * No explicit scope or RAII handle is required by the caller.
 */

#pragma once

#include <mutex>
#include <optional>
#include <string>

#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"

namespace nn
{

enum class DeviceType
{
    CPU,
    OPENCL,
};

struct Device
{
    DeviceType type = DeviceType::CPU; ///< Resolved device type.
    std::string id = "cpu";            ///< Canonical id string, e.g., "cpu" or "opencl".
    bool profiling_enabled = false;    ///< Enable OpenCL event profiling when true.

    /**
     * @brief Parse a device from a string token ("cpu", "opencl", "opencl:0", ...).
     * Unrecognised tokens fall back to CPU.
     */
    static auto from_string(const std::string& s) -> Device
    {
        if (s.empty()) return Device{DeviceType::CPU, "cpu"};
        if (s == "cpu") return Device{DeviceType::CPU, s};
        // Accept tokens like "opencl" or "opencl:0"
        if (s.rfind("opencl", 0) == 0) return Device{DeviceType::OPENCL, s};
        // Fallback to CPU
        return Device{DeviceType::CPU, s};
    }

    /// @brief Return a copy of this device with `profiling_enabled` set.
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
    std::string to_string() const
    {
        return id;
    }
};

/**
 * @brief Process-level lazy OpenCL runtime manager.
 *
 * `ensure_runtime()` is idempotent and thread-safe: the first call for an
 * OpenCL device starts the runtime and keeps it alive until program exit.
 * Subsequent calls are no-ops.  For CPU devices the call is always a no-op.
 *
 * This mirrors PyTorch's transparent device initialisation — callers do not
 * need to hold any RAII scope; runtime lifetime is managed through a
 * process-level static.
 */
struct DeviceRuntime
{
    static void ensure_runtime(const Device& device);
};

} // namespace nn
