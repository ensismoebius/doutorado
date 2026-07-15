/**
 * @file src/core/tensor/sycl/SYCLTensorBackend.cpp
 * @brief SYCL 2020 kernels for SYCLTensorBackend.
 *
 * Requires a SYCL implementation (AdaptiveCpp or oneAPI DPC++); compiled only
 * when NN_BACKEND=SYCL. All ops are copy-in/copy-out against the host mirror
 * (correctness-first — no device-resident state to go stale), but there is no
 * CPU fallback: if no SYCL device is available, every compute op throws.
 * cmake/SyclGpuCapabilityCheck.cmake refuses to configure this backend at all
 * on hardware known to make that likely (see its header comment for why);
 * this throw is the last-resort guard for whatever that static check misses.
 */

#include "tensor/sycl/SYCLTensorBackend.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <sycl/sycl.hpp>

#include "logging/Logger.hpp"

namespace
{

/// Either a working queue, or the error that prevented creating one. Queried
/// once (function-local static init) and cached — device probing is not
/// repeated on every call.
struct QueueOrError
{
    sycl::queue* queue = nullptr;
    std::string error;
};

const QueueOrError& global_queue_state()
{
    // Deliberately heap-allocated and never freed (exception to the
    // project's no-raw-new rule — see CLAUDE.md). A function-local *static*
    // sycl::queue gets torn down by a normal C++ static destructor at
    // process exit; under AdaptiveCpp's ROCm/generic backend on an
    // integrated GPU, that teardown has been observed to race with a
    // concurrently-exiting sibling process doing the same thing (e.g. two
    // ctest workers), reproducibly segfaulting inside the driver after the
    // test itself had already passed. Never destructing the queue means the
    // OS reclaims the GPU context at process exit instead of the runtime's
    // own (racy, under concurrency) destructor path.
    static QueueOrError state = []() -> QueueOrError
    {
        try
        {
            auto* q = new sycl::queue(sycl::default_selector_v, sycl::property::queue::in_order{});
            NN_LOG_INFO("SYCL backend: using device '" +
                        q->get_device().get_info<sycl::info::device::name>() + "'");
            return QueueOrError{q, {}};
        }
        catch (const std::exception& e)
        {
            return QueueOrError{nullptr, e.what()};
        }
    }();
    return state;
}

/// Returns the queue or throws — no CPU fallback. Every compute method routes
/// through this; there is no "if device unavailable, use m_host" branch
/// anywhere in this file by design (see file header).
sycl::queue& global_queue()
{
    const QueueOrError& state = global_queue_state();
    if (state.queue == nullptr)
        throw std::runtime_error("SYCL backend: no usable device (" + state.error +
                                 "). This backend has no CPU fallback by design — pick a different "
                                 "NN_BACKEND preset if this machine has no working SYCL device.");
    return *state.queue;
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
    // Status query only — does not throw, unlike global_queue(). Safe
    // because it substitutes no computation; it just answers a question.
    return global_queue_state().queue != nullptr;
}

std::string SYCLTensorBackend::device_description()
{
    const QueueOrError& state = global_queue_state();
    if (state.queue == nullptr) return "unavailable (" + state.error + ")";
    const auto dev = state.queue->get_device();
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

    sycl::queue& q = global_queue();
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

    sycl::queue& q = global_queue();
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

    sycl::queue& q = global_queue();
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

} // namespace

// ── Elementwise binary ───────────────────────────────────────────────────────
// No CPU fallback: global_queue() (called inside binary_op/unary_op/reduce_sum)
// throws if no device is available. See file header.

SYCLTensorBackend SYCLTensorBackend::add(const SYCLTensorBackend& other) const
{
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for add");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a + b; }));
}

SYCLTensorBackend SYCLTensorBackend::subtract(const SYCLTensorBackend& other) const
{
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for subtract");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a - b; }));
}

SYCLTensorBackend SYCLTensorBackend::multiply(const SYCLTensorBackend& other) const
{
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for multiply");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a * b; }));
}

SYCLTensorBackend SYCLTensorBackend::divide(const SYCLTensorBackend& other) const
{
    // Unlike XTensorBackend::divide (xtensor broadcasting), the device kernel
    // requires identical shapes — no production call site in this codebase
    // divides broadcast-shaped tensors (Tanh/Sigmoid/Adam/LSTM all divide
    // same-shaped operands); this is a real feature gap, not a fallback.
    if (shape() != other.shape()) throw std::invalid_argument("Shape mismatch for divide");
    return SYCLTensorBackend(
        binary_op(m_host, other.m_host, [](float a, float b) { return a / b; }));
}

void SYCLTensorBackend::add_inplace(const SYCLTensorBackend& other)
{
    *this = add(other);
}

void SYCLTensorBackend::subtract_inplace(const SYCLTensorBackend& other)
{
    *this = subtract(other);
}

void SYCLTensorBackend::multiply_inplace(const SYCLTensorBackend& other)
{
    *this = multiply(other);
}

void SYCLTensorBackend::divide_inplace(const SYCLTensorBackend& other)
{
    *this = divide(other);
}

// ── Scalar ops ───────────────────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::add_scalar(float val) const
{
    return SYCLTensorBackend(unary_op(m_host, [val](float a) { return a + val; }));
}

SYCLTensorBackend SYCLTensorBackend::multiply_scalar(float val) const
{
    return SYCLTensorBackend(unary_op(m_host, [val](float a) { return a * val; }));
}

SYCLTensorBackend SYCLTensorBackend::divide_scalar(float val) const
{
    return SYCLTensorBackend(unary_op(m_host, [val](float a) { return a / val; }));
}

void SYCLTensorBackend::add_scalar_inplace(float val)
{
    *this = add_scalar(val);
}

void SYCLTensorBackend::multiply_scalar_inplace(float val)
{
    *this = multiply_scalar(val);
}

void SYCLTensorBackend::divide_scalar_inplace(float val)
{
    *this = divide_scalar(val);
}

// ── Elementwise unary ────────────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::exp() const
{
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::exp(a); }));
}

SYCLTensorBackend SYCLTensorBackend::sqrt() const
{
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::sqrt(a); }));
}

SYCLTensorBackend SYCLTensorBackend::square() const
{
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return a * a; }));
}

SYCLTensorBackend SYCLTensorBackend::abs() const
{
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::fabs(a); }));
}

SYCLTensorBackend SYCLTensorBackend::relu() const
{
    return SYCLTensorBackend(unary_op(m_host, [](float a) { return sycl::fmax(a, 0.0f); }));
}

SYCLTensorBackend SYCLTensorBackend::leaky_relu(float alpha) const
{
    return SYCLTensorBackend(
        unary_op(m_host, [alpha](float a) { return a > 0.0f ? a : alpha * a; }));
}

void SYCLTensorBackend::sqrt_inplace()
{
    *this = sqrt();
}

void SYCLTensorBackend::square_inplace()
{
    *this = square();
}

// ── Matmul family / transpose ────────────────────────────────────────────────

SYCLTensorBackend SYCLTensorBackend::matmul(const SYCLTensorBackend& other) const
{
    if (shape().size() != 2 || other.shape().size() != 2)
        throw std::invalid_argument("Tensors must be 2D");
    if (cols() != other.rows()) throw std::invalid_argument("Dimension mismatch for matmul");

    const std::size_t M = rows(), K = cols(), N = other.cols();
    SYCLTensorBackend out(M, N);
    if (M == 0 || N == 0) return out;

    sycl::queue& q = global_queue();
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
    if (shape().size() != 2 || other.shape().size() != 2)
        throw std::invalid_argument("Tensors must be 2D");
    if (cols() != other.cols())
        throw std::invalid_argument("Dimension mismatch for matmul_transposed");

    // this (M,K) × otherᵀ (K,N) where other is stored (N,K) row-major.
    const std::size_t M = rows(), K = cols(), N = other.rows();
    SYCLTensorBackend out(M, N);
    if (M == 0 || N == 0) return out;

    sycl::queue& q = global_queue();
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
    if (shape().size() != 2) throw std::invalid_argument("Tensor must be 2D");

    const std::size_t R = rows(), C = cols();
    SYCLTensorBackend out(C, R);
    if (R == 0 || C == 0) return out;

    sycl::queue& q = global_queue();
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
    return reduce_sum(m_host, [](float a) { return a; });
}

float SYCLTensorBackend::norm() const
{
    return std::sqrt(reduce_sum(m_host, [](float a) { return a * a; }));
}

float SYCLTensorBackend::mean_squared_error(const SYCLTensorBackend& target) const
{
    if (size() != target.size())
        throw std::invalid_argument("Shape mismatch for mean_squared_error");
    const std::size_t n = size();
    if (n == 0) return 0.0f;

    sycl::queue& q = global_queue();
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
    const std::size_t R = rows(), C = cols();
    SYCLTensorBackend out(R, 1);
    if (R == 0 || C == 0)
    {
        out.set_zero();
        return out;
    }

    sycl::queue& q = global_queue();
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
    const std::size_t R = rows(), C = cols();
    SYCLTensorBackend out(1, C);
    if (R == 0 || C == 0)
    {
        out.set_zero();
        return out;
    }

    sycl::queue& q = global_queue();
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
