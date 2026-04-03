#ifndef DEVICE_TENSOR_BACKEND_HPP
#define DEVICE_TENSOR_BACKEND_HPP

#include <algorithm>

#include "nn/tensor/eigen/EigenTensorBackend.hpp"

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
 *
 */
class DeviceTensorBackend
{
   public:
    // Below you'll find concise, per-method annotations (allocation,
    // copy semantics, gradients, kernels) placed next to the methods
    // you will need to implement.

    // -- Constructors ----------------------------------------------------
    // Keep constructors lightweight. Avoid performing device allocations here;
    // prefer explicit `allocate_on_device()` to control allocation timing.
    DeviceTensorBackend() = default;

    // Construct a 2D host-backed tensor. Does not allocate device memory.
    explicit DeviceTensorBackend(Index rows, Index cols) : m_host(rows, cols) {}

    // Construct a 4D host-backed tensor (stored using the 4D->2D convention).
    // Note: avoid device allocations here; call `allocate_on_device()` when ready.
    explicit DeviceTensorBackend(Index d1, Index d2, Index d3, Index d4) : m_host(d1, d2, d3, d4) {}

    // Construct from arbitrary shape vector (may be >2 dimensions).
    // Ensure `reshape()` semantics are preserved on device copies.
    explicit DeviceTensorBackend(const std::vector<Index>& shape) : m_host(shape) {}

    // Construct directly from an Eigen matrix (host data). For device
    // backends prefer explicit host->device transfers via `copy_to_device()`.
    explicit DeviceTensorBackend(const Eigen::MatrixXf& data) : m_host(data) {}

    explicit DeviceTensorBackend(Eigen::MatrixXf&& data) : m_host(std::move(data)) {}

    // Copy / Move

    // Special members (copy / move): short, plain notes.

    // Copy constructor
    // Role: create a new object that is a copy of `other`.
    // Skeleton: performs member-wise copy of host and simulated device vectors
    // and duplicates the boolean flags.
    // Risk: copying raw device pointers (in a real backend) leads to double
    // free or use-after-free. Prefer one of:
    //  - Host-centric copy: copy only the host mirror and set device flags
    //    false so device memory is reallocated on demand.
    //  - Deep device copy: allocate device memory and copy device contents.
    // Example host-centric: DeviceTensorBackend(const DeviceTensorBackend& other)
    //   : m_host(other.m_host), m_on_device(false), m_grad_on_device(false) {}
    DeviceTensorBackend(const DeviceTensorBackend& other) = default;

    // Move constructor
    // Role: transfer resources from `other` to this object (leave `other`
    // valid but unspecified).
    // Keep `noexcept` if move cannot throw (helps containers). If your
    // backend cannot transfer device ownership, delete the move ctor.
    DeviceTensorBackend(DeviceTensorBackend&&) noexcept = default;

    // Copy assignment
    // Role: overwrite this object with a copy of `other`.
    // Same risks as copy constructor — avoid duplicating ownership of device
    // resources without proper bookkeeping.
    DeviceTensorBackend& operator=(const DeviceTensorBackend& other) = default;

    // Move assignment
    // Role: overwrite this object by moving resources from `other`.
    // Keep `noexcept` when safe; otherwise delete to enforce explicit policies.
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

    // Element-wise arithmetic (add/subtract/multiply).
    // - Implement device kernels for element-wise ops for performance.
    // - Ensure shape compatibility checks match `EigenTensorBackend` behaviour.

    void add_inplace(const DeviceTensorBackend& other)
    {
        m_host.add_inplace(other.m_host);
    }
    void subtract_inplace(const DeviceTensorBackend& other)
    {
        m_host.subtract_inplace(other.m_host);
    }
    void multiply_inplace(const DeviceTensorBackend& other)
    {
        m_host.multiply_inplace(other.m_host);
    }
    void divide_inplace(const DeviceTensorBackend& other)
    {
        m_host.divide_inplace(other.m_host);
    }
    void add_scalar_inplace(float val)
    {
        m_host.add_scalar_inplace(val);
    }
    void multiply_scalar_inplace(float val)
    {
        m_host.multiply_scalar_inplace(val);
    }
    void divide_scalar_inplace(float val)
    {
        m_host.divide_scalar_inplace(val);
    }
    void sqrt_inplace()
    {
        m_host.sqrt_inplace();
    }
    void square_inplace()
    {
        m_host.square_inplace();
    }

    DeviceTensorBackend exp() const
    {
        return DeviceTensorBackend(m_host.exp());
    }
    DeviceTensorBackend rowwise_sum() const
    {
        return DeviceTensorBackend(m_host.rowwise_sum());
    }
    DeviceTensorBackend matmul_transposed(const DeviceTensorBackend& other) const
    {
        return DeviceTensorBackend(m_host.matmul_transposed(other.m_host));
    }

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

    // Matrix multiply: implement a device-accelerated path (cuBLAS/hipBLAS)
    // when possible. Provide a host fallback for testing/portability.
    DeviceTensorBackend matmul(const DeviceTensorBackend& other) const
    {
        return DeviceTensorBackend(m_host.matmul(other.m_host));
    }

    // Transpose (2D only). Implement an in-device transpose when possible
    // to avoid a host round-trip. Keep API semantics identical to Eigen fallback.
    DeviceTensorBackend transpose() const
    {
        return DeviceTensorBackend(m_host.transpose());
    }

    // Scalar unary/binary ops. Prefer an in-device implementation to
    // reduce copies for hot paths (e.g., broadcasting a scalar onto a large
    // tensor). Fallback to host for correctness tests.
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

    // Element-wise unary functions. Implement device kernels where
    // practical. Ensure numerical stability and consistent edge-case handling
    // (NaN/Infs) with Eigen reference behaviour.
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

    // Reductions: consider using device-native reductions for performance
    // (e.g., thrust, custom kernels, or cuBLAS reductions). Keep a host
    // fallback for correctness tests.
    float mean_squared_error(const DeviceTensorBackend& target) const
    {
        return m_host.mean_squared_error(target.m_host);
    }
    // Reductions: norm and sum. For large tensors use device-native
    // reductions (parallel tree-reduce or library routines) to avoid host
    // bottlenecks. Keep a host fallback for verification and testing.
    float norm() const
    {
        return m_host.norm();
    }
    float sum() const
    {
        return m_host.sum();
    }

    // Row/column reductions return small tensors; consider computing them
    // on-device and returning host copies as needed.
    DeviceTensorBackend sum_rows() const
    {
        return DeviceTensorBackend(m_host.sum_rows());
    }
    DeviceTensorBackend sum_cols() const
    {
        return DeviceTensorBackend(m_host.sum_cols());
    }

    // NaN detection helper; for device backends consider performing a
    // short-circuit device scan to avoid copying entire tensors to host.
    bool hasNaN() const
    {
        return m_host.hasNaN();
    }

    // Slicing / Blocks
    // Views / slicing - these return *copies* in the current design.
    // For device backends you can implement zero-copy views if you expose
    // a proper view type. Otherwise, implement device-side copying and
    // document the cost of these operations.
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

    // Mutators: update both host mirror and optionally device buffer.
    // - For device backends, either perform the operation on-device or
    //   update host mirror and mark device buffer stale/dirty.
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

    // Raw data pointer semantics (Step 6 in BackendGuide.md):
    // - For device backends decide whether `data_ptr()` returns a host-accessible
    //   pointer (in which case you must ensure `copy_to_host()` happened) or
    //   if it returns nullptr / throws when data is device-only. Provide
    //   `device_data_ptr()` to expose device pointers explicitly.
    const float* data_ptr() const
    {
        return m_host.data_ptr();
    }
    float* mutable_data_ptr()
    {
        return m_host.mutable_data_ptr();
    }

    // Gradient API (Step 6 in BackendGuide.md):
    // - Decide and document whether `get_grad()` performs an implicit host
    //   copy from device memory or requires an explicit `grad_to_host()` call.
    // - If gradients live on-device by default, provide `grad_to_host()` and
    //   `grad_to_device()` helpers and keep `get_grad()` as an explicit host-copy.
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

    // Return a mutable reference to gradient storage. If gradients are
    // stored on-device in your implementation, ensure appropriate host-device
    // synchronization before returning a host-visible grad reference.
    DeviceTensorBackend& grad_ref()
    {
        m_host.grad_ref();
        return *this;
    }

    // Equality compares host mirrors. If your backend keeps device-only
    // state, ensure you copy/sync device data to host before comparing, or
    // override these operators to compare device buffers directly when safe.
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

    // -----------------------------------------------------------------
    // Gradient device-mirror helpers (for testing and as a reference for
    // real backends). These mirror the host gradient into a device buffer
    // and back. Replace with device-native implementations when available.
    // -----------------------------------------------------------------
    void allocate_device_grad()
    {
        m_device_grad.assign(static_cast<size_t>(size()), 0.0f);
        m_grad_on_device = true;
    }

    void copy_grad_to_device()
    {
        // Copy the host gradient into the device grad buffer. The host grad is
        // obtained via m_host.get_grad() which returns an EigenTensorBackend
        // value; copy its contents into the device vector.
        if (!m_grad_on_device) allocate_device_grad();
        auto host_grad = m_host.get_grad();
        const float* src = host_grad.data_ptr();
        std::copy(src, src + static_cast<size_t>(size()), m_device_grad.begin());
        m_grad_on_device = true;
    }

    void copy_grad_to_host()
    {
        // Copy device grad back into host gradient storage. Ensure host grad
        // storage exists by calling grad_ref() which lazily allocates it.
        if (!m_grad_on_device) return;
        auto& host_grad_ref = m_host.grad_ref();
        std::copy(m_device_grad.begin(),
            m_device_grad.begin() + static_cast<size_t>(size()),
            host_grad_ref.mutable_data_ptr());
    }

    bool is_grad_on_device() const
    {
        return m_grad_on_device;
    }

    float* mutable_device_grad_ptr()
    {
        if (!m_grad_on_device) allocate_device_grad();
        return m_device_grad.data();
    }

    const float* device_grad_ptr() const
    {
        return m_grad_on_device ? m_device_grad.data() : nullptr;
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
    // concrete implementations. Use this as a reliable host fallback in tests.
    EigenTensorBackend m_host;

    // Simulated device buffer (for the skeleton): stores a host-side copy of
    // what would be device memory. Replace with an actual device pointer,
    // allocator, and optional stream/context state in real backends. Ensure
    // you track allocation size and alignment when using device allocators.
    std::vector<float> m_device;
    bool m_on_device = false;

    // Simulated device gradient mirror (for testing).
    // Replace with device-native gradient storage in real backends.
    std::vector<float> m_device_grad;
    bool m_grad_on_device = false;
};

} // namespace nn

#endif // DEVICE_TENSOR_BACKEND_HPP
