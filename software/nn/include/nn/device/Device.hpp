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

/**
 * @brief Lightweight device descriptor (value type, PyTorch-like).
 *
 * Carries the device type, an identifier string, and optional per-device
 * settings. Created via `from_string()` and passed to `Module::to()` or
 * `DeviceRuntime::ensure_runtime()`.
 */
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
    /**
     * @brief Lazily initialise the device runtime (thread-safe, idempotent).
     *
     * For OpenCL: starts the runtime and buffer pool on the first call, using
     * the `profiling_enabled` flag from `device`.  For CPU: always a no-op.
     *
     * @throws std::runtime_error when OpenCL initialisation fails.
     */
    static void ensure_runtime(const Device& device)
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
};

} // namespace nn
