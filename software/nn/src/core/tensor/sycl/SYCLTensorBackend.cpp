/**
 * @file src/core/tensor/sycl/SYCLTensorBackend.cpp
 * @brief SYCL 2020 kernels for SYCLTensorBackend.
 *
 * Requires a SYCL implementation (AdaptiveCpp or oneAPI DPC++); compiled only
 * when NN_BACKEND=SYCL. All ops are copy-in/copy-out against the host mirror:
 * correctness-first — no device-resident state to go stale. Every op falls
 * back to the XTensorBackend host implementation when no SYCL device exists.
 */

#include "tensor/sycl/SYCLTensorBackend.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include "logging/Logger.hpp"

namespace
{

/// Process-wide in-order queue; nullptr when no device could be initialized.
sycl::queue* global_queue()
{
    static sycl::queue* queue = []() -> sycl::queue*
    {
        try
        {
            static sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
            NN_LOG_INFO("SYCL backend: using device '" +
                        q.get_device().get_info<sycl::info::device::name>() + "'");
            return &q;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("SYCL backend: no device available (") + e.what() +
                        "), falling back to host execution");
            return nullptr;
        }
    }();
    return queue;
}

/// RAII USM device buffer. Never throws on free.
class DevBuf
{
   public:
    DevBuf(sycl::queue& q, std::size_t n) : m_q(q), m_p(sycl::malloc_device<float>(n, q))
    {
        if (m_p == nullptr) throw std::runtime_error("SYCL backend: device allocation failed");
    }
    ~DevBuf()
    {
        sycl::free(m_p, m_q);
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;

    float* get() const
    {
        return m_p;
    }

   private:
    sycl::queue& m_q;
    float* m_p;
};

} // namespace

namespace nn
{

bool SYCLTensorBackend::sycl_runtime_available()
{
    return global_queue() != nullptr;
}

std::string SYCLTensorBackend::device_description()
{
    sycl::queue* q = global_queue();
    if (q == nullptr) return "host fallback";
    const auto dev = q->get_device();
    return dev.get_info<sycl::info::device::vendor>() + " " +
           dev.get_info<sycl::info::device::name>();
}

namespace
{

/// Copy-in / kernel / copy-out driver for elementwise binary ops.
/// `Op` is a device-callable functor: float(float a, float b).
template <typename Op>
XTensorBackend binary_op(const XTensorBackend& a, const XTensorBackend& b, Op op)
{
    XTensorBackend out(a.shape());
    const std::size_t n = a.size();
    if (n == 0) return out;

    sycl::queue& q = *global_queue();
    DevBuf da(q, n), db(q, n), dr(q, n);
    q.memcpy(da.get(), a.data_ptr(), n * sizeof(float));
    q.memcpy(db.get(), b.data_ptr(), n * sizeof(float));
    const float* pa = da.get();
    const float* pb = db.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { pr[i] = op(pa[i], pb[i]); });
    q.memcpy(out.mutable_data_ptr(), dr.get(), n * sizeof(float)).wait();
    return out;
}

/// Copy-in / kernel / copy-out driver for elementwise unary ops.
template <typename Op>
XTensorBackend unary_op(const XTensorBackend& a, Op op)
{
    XTensorBackend out(a.shape());
    const std::size_t n = a.size();
    if (n == 0) return out;

    sycl::queue& q = *global_queue();
    DevBuf da(q, n), dr(q, n);
    q.memcpy(da.get(), a.data_ptr(), n * sizeof(float));
    const float* pa = da.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { pr[i] = op(pa[i]); });
    q.memcpy(out.mutable_data_ptr(), dr.get(), n * sizeof(float)).wait();
    return out;
}

/// Device sum-reduction over an arbitrary per-element transform.
/// `Xf` maps (a[i]) -> float; result is Σ Xf(a[i]).
template <typename Xf>
float reduce_sum(const XTensorBackend& a, Xf xf)
{
    const std::size_t n = a.size();
    if (n == 0) return 0.0f;

    sycl::queue& q = *global_queue();
    DevBuf da(q, n), dsum(q, 1);
    q.memcpy(da.get(), a.data_ptr(), n * sizeof(float));
    q.memset(dsum.get(), 0, sizeof(float));
    const float* pa = da.get();
    float* psum = dsum.get();
    q.parallel_for(sycl::range<1>(n),
        sycl::reduction(psum, sycl::plus<float>{}),
        [=](sycl::id<1> i, auto& acc) { acc += xf(pa[i]); });
    float result = 0.0f;
    q.memcpy(&result, dsum.get(), sizeof(float)).wait();
    return result;
}

bool device_ready()
{
    return global_queue() != nullptr;
}

} // namespace

// ── Elementwise binary ───────────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::add(const SYCLTensorBackend& other) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.add(other.m_host));
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for add");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a + b; }));
}

SYCLTensorBackend SYCLTensorBackend::subtract(const SYCLTensorBackend& other) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.subtract(other.m_host));
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for subtract");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a - b; }));
}

SYCLTensorBackend SYCLTensorBackend::multiply(const SYCLTensorBackend& other) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.multiply(other.m_host));
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for multiply");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a * b; }));
}

SYCLTensorBackend SYCLTensorBackend::divide(const SYCLTensorBackend& other) const
{
    // XTensorBackend::divide performs no shape check (xtensor broadcasting);
    // divergent shapes are rare and broadcasting is host-mirror territory.
    if (!device_ready() || shape() != other.shape())
        return SYCLTensorBackend(m_host.divide(other.m_host));
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a / b; }));
}

void SYCLTensorBackend::add_inplace(const SYCLTensorBackend& other)
{
    if (!device_ready())
    {
        m_host.add_inplace(other.m_host);
        return;
    }
    *this = add(other);
}

void SYCLTensorBackend::subtract_inplace(const SYCLTensorBackend& other)
{
    if (!device_ready())
    {
        m_host.subtract_inplace(other.m_host);
        return;
    }
    *this = subtract(other);
}

void SYCLTensorBackend::multiply_inplace(const SYCLTensorBackend& other)
{
    if (!device_ready())
    {
        m_host.multiply_inplace(other.m_host);
        return;
    }
    *this = multiply(other);
}

void SYCLTensorBackend::divide_inplace(const SYCLTensorBackend& other)
{
    if (!device_ready())
    {
        m_host.divide_inplace(other.m_host);
        return;
    }
    *this = divide(other);
}

// ── Scalar ops ───────────────────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::add_scalar(float val) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.add_scalar(val));
    return SYCLTensorBackend(unary_op(m_host, [val](float a) { return a + val; }));
}

SYCLTensorBackend SYCLTensorBackend::multiply_scalar(float val) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.multiply_scalar(val));
    return SYCLTensorBackend(unary_op(m_host, [val](float a) { return a * val; }));
}

SYCLTensorBackend SYCLTensorBackend::divide_scalar(float val) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.divide_scalar(val));
    return SYCLTensorBackend(unary_op(m_host, [val](float a) { return a / val; }));
}

void SYCLTensorBackend::add_scalar_inplace(float val)
{
    if (!device_ready())
    {
        m_host.add_scalar_inplace(val);
        return;
    }
    *this = add_scalar(val);
}

void SYCLTensorBackend::multiply_scalar_inplace(float val)
{
    if (!device_ready())
    {
        m_host.multiply_scalar_inplace(val);
        return;
    }
    *this = multiply_scalar(val);
}

void SYCLTensorBackend::divide_scalar_inplace(float val)
{
    if (!device_ready())
    {
        m_host.divide_scalar_inplace(val);
        return;
    }
    *this = divide_scalar(val);
}

// ── Elementwise unary ────────────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::exp() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.exp());
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::exp(a); }));
}

SYCLTensorBackend SYCLTensorBackend::sqrt() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.sqrt());
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::sqrt(a); }));
}

SYCLTensorBackend SYCLTensorBackend::square() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.square());
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return a * a; }));
}

SYCLTensorBackend SYCLTensorBackend::abs() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.abs());
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::fabs(a); }));
}

SYCLTensorBackend SYCLTensorBackend::relu() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.relu());
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::fmax(a, 0.0f); }));
}

SYCLTensorBackend SYCLTensorBackend::leaky_relu(float alpha) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.leaky_relu(alpha));
    return SYCLTensorBackend(
        unary_op(m_host, [alpha](float a) { return a > 0.0f ? a : alpha * a; }));
}

void SYCLTensorBackend::sqrt_inplace()
{
    if (!device_ready())
    {
        m_host.sqrt_inplace();
        return;
    }
    *this = sqrt();
}

void SYCLTensorBackend::square_inplace()
{
    if (!device_ready())
    {
        m_host.square_inplace();
        return;
    }
    *this = square();
}

// ── Matmul family / transpose ────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::matmul(const SYCLTensorBackend& other) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.matmul(other.m_host));
    if (shape().size() != 2 || other.shape().size() != 2)
        throw std::invalid_argument("Tensors must be 2D");
    if (cols() != other.rows()) throw std::invalid_argument("Dimension mismatch for matmul");

    const std::size_t M = rows(), K = cols(), N = other.cols();
    SYCLTensorBackend out(M, N);
    if (M == 0 || N == 0) return out;

    sycl::queue& q = *global_queue();
    DevBuf da(q, M * K), db(q, K * N), dr(q, M * N);
    q.memcpy(da.get(), data_ptr(), M * K * sizeof(float));
    q.memcpy(db.get(), other.data_ptr(), K * N * sizeof(float));
    const float* pa = da.get();
    const float* pb = db.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<2>(M, N),
        [=](sycl::id<2> idx)
        {
            const std::size_t i = idx[0], j = idx[1];
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) acc += pa[i * K + k] * pb[k * N + j];
            pr[i * N + j] = acc;
        });
    q.memcpy(out.mutable_data_ptr(), dr.get(), M * N * sizeof(float)).wait();
    return out;
}

SYCLTensorBackend SYCLTensorBackend::matmul_transposed(const SYCLTensorBackend& other) const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.matmul_transposed(other.m_host));
    if (shape().size() != 2 || other.shape().size() != 2)
        throw std::invalid_argument("Tensors must be 2D");
    if (cols() != other.cols())
        throw std::invalid_argument("Dimension mismatch for matmul_transposed");

    // this (M,K) × otherᵀ (K,N) where other is stored (N,K) row-major.
    const std::size_t M = rows(), K = cols(), N = other.rows();
    SYCLTensorBackend out(M, N);
    if (M == 0 || N == 0) return out;

    sycl::queue& q = *global_queue();
    DevBuf da(q, M * K), db(q, N * K), dr(q, M * N);
    q.memcpy(da.get(), data_ptr(), M * K * sizeof(float));
    q.memcpy(db.get(), other.data_ptr(), N * K * sizeof(float));
    const float* pa = da.get();
    const float* pb = db.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<2>(M, N),
        [=](sycl::id<2> idx)
        {
            const std::size_t i = idx[0], j = idx[1];
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) acc += pa[i * K + k] * pb[j * K + k];
            pr[i * N + j] = acc;
        });
    q.memcpy(out.mutable_data_ptr(), dr.get(), M * N * sizeof(float)).wait();
    return out;
}

SYCLTensorBackend SYCLTensorBackend::transpose() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.transpose());
    if (shape().size() != 2) throw std::invalid_argument("Tensor must be 2D");

    const std::size_t R = rows(), C = cols();
    SYCLTensorBackend out(C, R);
    if (R == 0 || C == 0) return out;

    sycl::queue& q = *global_queue();
    DevBuf da(q, R * C), dr(q, R * C);
    q.memcpy(da.get(), data_ptr(), R * C * sizeof(float));
    const float* pa = da.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<2>(R, C),
        [=](sycl::id<2> idx) { pr[idx[1] * R + idx[0]] = pa[idx[0] * C + idx[1]]; });
    q.memcpy(out.mutable_data_ptr(), dr.get(), R * C * sizeof(float)).wait();
    return out;
}

// ── Reductions ───────────────────────────────────────────────────────────────

float SYCLTensorBackend::sum() const
{
    if (!device_ready()) return m_host.sum();
    return reduce_sum(m_host, [](float a) { return a; });
}

float SYCLTensorBackend::norm() const
{
    if (!device_ready()) return m_host.norm();
    return std::sqrt(reduce_sum(m_host, [](float a) { return a * a; }));
}

float SYCLTensorBackend::mean_squared_error(const SYCLTensorBackend& target) const
{
    if (!device_ready()) return m_host.mean_squared_error(target.m_host);
    if (size() != target.size())
        throw std::invalid_argument("Shape mismatch for mean_squared_error");
    const std::size_t n = size();
    if (n == 0) return 0.0f;

    sycl::queue& q = *global_queue();
    DevBuf da(q, n), db(q, n), dsum(q, 1);
    q.memcpy(da.get(), data_ptr(), n * sizeof(float));
    q.memcpy(db.get(), target.data_ptr(), n * sizeof(float));
    q.memset(dsum.get(), 0, sizeof(float));
    const float* pa = da.get();
    const float* pb = db.get();
    float* psum = dsum.get();
    q.parallel_for(sycl::range<1>(n),
        sycl::reduction(psum, sycl::plus<float>{}),
        [=](sycl::id<1> i, auto& acc)
        {
            const float d = pa[i] - pb[i];
            acc += d * d;
        });
    float sq = 0.0f;
    q.memcpy(&sq, dsum.get(), sizeof(float)).wait();
    return sq / static_cast<float>(n);
}

SYCLTensorBackend SYCLTensorBackend::sum_rows() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.sum_rows());
    const std::size_t R = rows(), C = cols();
    SYCLTensorBackend out(R, 1);
    if (R == 0 || C == 0)
    {
        out.set_zero();
        return out;
    }

    sycl::queue& q = *global_queue();
    DevBuf da(q, R * C), dr(q, R);
    q.memcpy(da.get(), data_ptr(), R * C * sizeof(float));
    const float* pa = da.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<1>(R),
        [=](sycl::id<1> i)
        {
            float acc = 0.0f;
            for (std::size_t j = 0; j < C; ++j) acc += pa[i * C + j];
            pr[i] = acc;
        });
    q.memcpy(out.mutable_data_ptr(), dr.get(), R * sizeof(float)).wait();
    return out;
}

SYCLTensorBackend SYCLTensorBackend::sum_cols() const
{
    if (!device_ready()) return SYCLTensorBackend(m_host.sum_cols());
    const std::size_t R = rows(), C = cols();
    SYCLTensorBackend out(1, C);
    if (R == 0 || C == 0)
    {
        out.set_zero();
        return out;
    }

    sycl::queue& q = *global_queue();
    DevBuf da(q, R * C), dr(q, C);
    q.memcpy(da.get(), data_ptr(), R * C * sizeof(float));
    const float* pa = da.get();
    float* pr = dr.get();
    q.parallel_for(sycl::range<1>(C),
        [=](sycl::id<1> j)
        {
            float acc = 0.0f;
            for (std::size_t i = 0; i < R; ++i) acc += pa[i * C + j];
            pr[j] = acc;
        });
    q.memcpy(out.mutable_data_ptr(), dr.get(), C * sizeof(float)).wait();
    return out;
}

SYCLTensorBackend SYCLTensorBackend::rowwise_sum() const
{
    return sum_rows();
}

} // namespace nn
