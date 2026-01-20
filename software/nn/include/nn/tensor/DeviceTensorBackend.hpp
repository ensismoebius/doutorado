#ifndef DEVICE_TENSOR_BACKEND_HPP
#define DEVICE_TENSOR_BACKEND_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include "nn/tensor/EigenTensorBackend.hpp"

namespace nn
{

/**
 * @brief Skeleton "device" backend demonstrating how to implement another
 * tensor backend.
 *
 * This minimal example delegates all work to an internal Eigen-based host
 * mirror (m_host). It is intentionally simple so contributors can easily
 * replace the internals with device allocations (CUDA, ROCm) and implement
 * explicit copy_to_device()/copy_to_host() semantics.
 *
 * Required behaviour (contract): see `EigenTensorBackend.hpp` for detailed
 * expectations and per-method documentation. This skeleton preserves that
 * contract while showing where device-specific code would go.
 */
class DeviceTensorBackend
{
   public:
    // -- Constructors ----------------------------------------------------
    DeviceTensorBackend() = default;

    explicit DeviceTensorBackend(Index rows, Index cols) : m_host(rows, cols) {}

    explicit DeviceTensorBackend(Index d1, Index d2, Index d3, Index d4) : m_host(d1, d2, d3, d4) {}

    explicit DeviceTensorBackend(const std::vector<Index>& shape) : m_host(shape) {}

    explicit DeviceTensorBackend(const Eigen::MatrixXf& data) : m_host(data) {}

    explicit DeviceTensorBackend(Eigen::MatrixXf&& data) : m_host(std::move(data)) {}

    // Copy / Move
    DeviceTensorBackend(const DeviceTensorBackend& other) = default;
    DeviceTensorBackend(DeviceTensorBackend&&) noexcept = default;
    DeviceTensorBackend& operator=(const DeviceTensorBackend& other) = default;
    DeviceTensorBackend& operator=(DeviceTensorBackend&&) noexcept = default;

    // Static factories
    static DeviceTensorBackend zeros(Index rows, Index cols)
    {
        return DeviceTensorBackend(EigenTensorBackend::zeros(rows, cols));
    }

    static DeviceTensorBackend ones(Index rows, Index cols)
    {
        return DeviceTensorBackend(EigenTensorBackend::ones(rows, cols));
    }

    // Construct from an Eigen host backend directly. This is useful for
    // factories that produce `EigenTensorBackend` values; prefer this over
    // accessing private internals like `.m_data`.
    explicit DeviceTensorBackend(const EigenTensorBackend& host) : m_host(host) {}

    explicit DeviceTensorBackend(EigenTensorBackend&& host) : m_host(std::move(host)) {}

    // Shape / sizing
    const std::vector<Index>& shape() const
    {
        return m_host.shape();
    }
    void reshape(const std::vector<Index>& new_shape)
    {
        m_host.reshape(new_shape);
    }
    Index rows() const
    {
        return m_host.rows();
    }
    Index cols() const
    {
        return m_host.cols();
    }
    Index size() const
    {
        return m_host.size();
    }

    // Element access (delegates to host mirror)
    float& at(Index i)
    {
        return m_host.at(i);
    }
    const float& at(Index i) const
    {
        return m_host.at(i);
    }

    float& at(Index r, Index c)
    {
        return m_host.at(r, c);
    }
    const float& at(Index r, Index c) const
    {
        return m_host.at(r, c);
    }

    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        return m_host.at(d1, d2, d3, d4);
    }
    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        return m_host.at(d1, d2, d3, d4);
    }

    float& at(const std::vector<Index>& idx)
    {
        return m_host.at(idx);
    }
    const float& at(const std::vector<Index>& idx) const
    {
        return m_host.at(idx);
    }

    // Arithmetic ops (forward to host mirror)
    DeviceTensorBackend add(const DeviceTensorBackend& other) const
    {
        return DeviceTensorBackend(m_host.add(other.m_host));
    }
    DeviceTensorBackend subtract(const DeviceTensorBackend& other) const
    {
        return DeviceTensorBackend(m_host.subtract(other.m_host));
    }
    DeviceTensorBackend multiply(const DeviceTensorBackend& other) const
    {
        return DeviceTensorBackend(m_host.multiply(other.m_host));
    }

    DeviceTensorBackend matmul(const DeviceTensorBackend& other) const
    {
        return DeviceTensorBackend(m_host.matmul(other.m_host));
    }

    DeviceTensorBackend transpose() const
    {
        return DeviceTensorBackend(m_host.transpose());
    }

    DeviceTensorBackend add_scalar(float val) const
    {
        return DeviceTensorBackend(m_host.add_scalar(val));
    }
    DeviceTensorBackend multiply_scalar(float val) const
    {
        return DeviceTensorBackend(m_host.multiply_scalar(val));
    }
    DeviceTensorBackend divide_scalar(float val) const
    {
        return DeviceTensorBackend(m_host.divide_scalar(val));
    }

    DeviceTensorBackend sqrt() const
    {
        return DeviceTensorBackend(m_host.sqrt());
    }
    DeviceTensorBackend square() const
    {
        return DeviceTensorBackend(m_host.square());
    }
    DeviceTensorBackend abs() const
    {
        return DeviceTensorBackend(m_host.abs());
    }

    DeviceTensorBackend relu() const
    {
        return DeviceTensorBackend(m_host.relu());
    }
    DeviceTensorBackend leaky_relu(float alpha) const
    {
        return DeviceTensorBackend(m_host.leaky_relu(alpha));
    }

    // Reductions
    float mean_squared_error(const DeviceTensorBackend& target) const
    {
        return m_host.mean_squared_error(target.m_host);
    }
    float norm() const
    {
        return m_host.norm();
    }
    float sum() const
    {
        return m_host.sum();
    }

    DeviceTensorBackend sum_rows() const
    {
        return DeviceTensorBackend(m_host.sum_rows());
    }
    DeviceTensorBackend sum_cols() const
    {
        return DeviceTensorBackend(m_host.sum_cols());
    }

    bool hasNaN() const
    {
        return m_host.hasNaN();
    }

    // Slicing / Blocks
    DeviceTensorBackend row(Index i) const
    {
        return DeviceTensorBackend(m_host.row(i));
    }
    DeviceTensorBackend col(Index j) const
    {
        return DeviceTensorBackend(m_host.col(j));
    }
    DeviceTensorBackend leftCols(Index n) const
    {
        return DeviceTensorBackend(m_host.leftCols(n));
    }
    DeviceTensorBackend topRows(Index n) const
    {
        return DeviceTensorBackend(m_host.topRows(n));
    }
    DeviceTensorBackend block(Index r, Index c, Index rows, Index cols) const
    {
        return DeviceTensorBackend(m_host.block(r, c, rows, cols));
    }
    void setBlock(Index r, Index c, const DeviceTensorBackend& other)
    {
        m_host.setBlock(r, c, other.m_host);
    }
    DeviceTensorBackend slice(std::span<const int> indices) const
    {
        return DeviceTensorBackend(m_host.slice(indices));
    }

    // Mutators
    void fill(float v)
    {
        m_host.fill(v);
    }
    void set_zero()
    {
        m_host.set_zero();
    }
    void set_ones()
    {
        m_host.set_ones();
    }

    const float* data_ptr() const
    {
        return m_host.data_ptr();
    }
    float* mutable_data_ptr()
    {
        return m_host.mutable_data_ptr();
    }

    // Gradient API (delegated to host mirror; device backends should implement
    // device gradient handling and host/device synchronization as needed)
    DeviceTensorBackend get_grad() const
    {
        // Return the host gradient as a DeviceTensorBackend value. For real
        // device backends this may involve a host copy of device-resident
        // gradients; document your chosen strategy clearly.
        return DeviceTensorBackend(m_host.get_grad());
    }
    void set_grad(const DeviceTensorBackend& other)
    {
        m_host.set_grad(other.m_host);
    }
    void zero_grad()
    {
        m_host.zero_grad();
    }
    DeviceTensorBackend& grad_ref()
    {
        m_host.grad_ref();
        return *this;
    }

    bool operator==(const DeviceTensorBackend& other) const
    {
        return m_host == other.m_host;
    }
    bool operator!=(const DeviceTensorBackend& other) const
    {
        return !(*this == other);
    }

    // Device-specific helpers (simulated device buffer for tests)
    // - These methods provide a minimal, testable device-mock that can be
    //   replaced with real device allocation / copy logic for CUDA/ROCm.
    void allocate_on_device()
    {
        // Allocate a device-side buffer and mark as present.
        m_device.assign(static_cast<size_t>(size()), 0.0f);
        m_on_device = true;
    }

    void copy_to_device()
    {
        // Copy host->device. Allocates device buffer if not already present.
        if (!m_on_device) allocate_on_device();
        std::copy(
            m_host.data_ptr(), m_host.data_ptr() + static_cast<size_t>(size()), m_device.begin());
        m_on_device = true;
    }

    void copy_to_host()
    {
        // Copy device->host. No-op if device buffer absent.
        if (!m_on_device) return;
        std::copy(m_device.begin(),
                  m_device.begin() + static_cast<size_t>(size()),
                  m_host.mutable_data_ptr());
        // Keep device buffer allocated for potential reuse.
    }

    // Query whether a device buffer exists (useful for tests and debugging).
    bool is_on_device() const
    {
        return m_on_device;
    }

    // Access device-side memory for tests / mocks. For real device backends
    // these would return device pointers or perform explicit host-device transfers.
    float* mutable_device_data_ptr()
    {
        if (!m_on_device) allocate_on_device();
        return m_device.data();
    }
    const float* device_data_ptr() const
    {
        return m_on_device ? m_device.data() : nullptr;
    }

   private:
    // Host mirror (Eigen implementation). Replace with device-native storage in
    // concrete implementations.
    EigenTensorBackend m_host;

    // Simulated device buffer (for the skeleton): stores a host-side copy of
    // what would be device memory. Real backends would manage device memory
    // and provide proper synchronization primitives.
    std::vector<float> m_device;
    bool m_on_device = false;
};

} // namespace nn

#endif // DEVICE_TENSOR_BACKEND_HPP
