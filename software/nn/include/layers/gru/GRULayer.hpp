#ifndef NN_LAYERS_GRU_GRULAYER_HPP
#define NN_LAYERS_GRU_GRULAYER_HPP

/**
 * @file include/layers/gru/GRULayer.hpp
 * @brief Single-layer GRU cell with full BPTT and batch gradient accumulation.
 *
 * Gate equations per time step t (reset applied AFTER the recurrent matmul —
 * the cuDNN / PyTorch convention [1, 2]):
 *   pre^x_t = x_t W^T                              // (B, 3H)
 *   pre^h_t = H_{t-1} U^T                          // (B, 3H)
 *   r_t = sigma( pre^x_t[0:H]   + pre^h_t[0:H]   + b[0:H]   )   // reset gate
 *   z_t = sigma( pre^x_t[H:2H]  + pre^h_t[H:2H]  + b[H:2H]  )   // update gate
 *   n_t = tanh ( pre^x_t[2H:3H] + b[2H:3H] + r_t ⊙ pre^h_t[2H:3H] ) // candidate
 *   H_t = (1 - z_t) ⊙ n_t + z_t ⊙ H_{t-1}
 *
 * Gate ordering [r|z|n]; weights stored as stacked matrices in that order.
 * A single bias vector per gate (b_ih and b_hh of the reference are folded into
 * one; the candidate gate's bias sits on the input side so it is NOT scaled by
 * r_t). This layer is its own correctness reference and is validated by a
 * finite-difference gradient check (gru_layer_gtest.cpp).
 *
 * References:
 *   [1] Cho et al., EMNLP 2014. arXiv:1406.1078
 *   [2] Chung et al., NeurIPS DL Workshop 2014. arXiv:1412.3555
 *
 * Shape contract (identical to LSTMLayer):
 *   forward(Tensor{T, D})    -> Tensor{T, H}
 *   forward(Tensor{B, T, D}) -> Tensor{B, T, H}
 *   backward(Tensor{T, H})   -> Tensor{T, D}
 *   backward(Tensor{B, T, H}) -> Tensor{B, T, D}   (grads W/U/b accumulated over B)
 */

#include <cmath>
#include <map>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "layers/activations/FastActivations.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

template <typename Backend>
struct GRUStepCacheT
{
    using Tensor = nn::TensorImpl<Backend>;
    Tensor x;      // x_t                       (B, D)
    Tensor h_prev; // H_{t-1}                   (B, H)
    Tensor r;      // reset gate                (B, H)
    Tensor z;      // update gate               (B, H)
    Tensor n;      // candidate                 (B, H)
    Tensor uh_n;   // (H_{t-1} U^T)[:, 2H:3H]   (B, H) — the reset-gated recurrent term
};

template <typename Backend>
class GRULayerImpl : public Module<Backend>
{
   public:
    using Tensor = nn::TensorImpl<Backend>;

    int input_size_;
    int hidden_size_;

    /// true (DEFAULT) = exact sigmoid/tanh (PyTorch parity). false = the rational
    /// approximations in FastActivations.hpp. Backward picks the matching derivative.
    bool exact_activations = true;

    Tensor W_; // (3H, D)
    Tensor U_; // (3H, H)
    Tensor b_; // (3H, 1)

    Tensor dW_;
    Tensor dU_;
    Tensor db_;

    std::vector<Tensor*> param_ptrs_;

    Tensor h0_; // persistent hidden state for the 2-D single-sequence path

    std::vector<GRUStepCacheT<Backend>> cache_;
    bool requires_grad_ = false;

    explicit GRULayerImpl(int input_size, int hidden_size)
        : input_size_(input_size),
          hidden_size_(hidden_size),
          W_(3 * hidden_size, input_size),
          U_(3 * hidden_size, hidden_size),
          b_(3 * hidden_size, 1),
          dW_(3 * hidden_size, input_size),
          dU_(3 * hidden_size, hidden_size),
          db_(3 * hidden_size, 1),
          h0_(1, hidden_size)
    {
        auto normal_fill = [](Tensor& t, unsigned seed_offset)
        {
            std::mt19937 rng(42u + seed_offset);
            std::normal_distribution<float> dist(0.0f, 0.05f);
            for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k) t.at(k) = dist(rng);
        };

        normal_fill(W_, 0u);
        normal_fill(U_, 1u);
        b_.set_zero();
        h0_.set_zero();
        dW_.set_zero();
        dU_.set_zero();
        db_.set_zero();

        param_ptrs_ = {&W_, &U_, &b_};
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        const auto& shape = input.get_shape();
        const bool is_2d = (shape.size() == 2);

        int B = 1;
        Tensor seq_3d;
        if (is_2d)
            seq_3d = input.reshape({1, shape[0], shape[1]});
        else
        {
            B = static_cast<int>(shape[0]);
            seq_3d = input;
        }

        const int T_seq = static_cast<int>(seq_3d.get_shape()[1]);
        const int D_in = static_cast<int>(seq_3d.get_shape()[2]);
        if (D_in != input_size_)
            throw std::invalid_argument("GRULayerImpl::forward: input D=" + std::to_string(D_in) +
                                        " != input_size=" + std::to_string(input_size_));

        if (requires_grad)
        {
            cache_.clear();
            cache_.reserve(static_cast<std::size_t>(T_seq));
        }

        const nn::Index H = static_cast<nn::Index>(hidden_size_);
        const nn::Index Bi = static_cast<nn::Index>(B);

        Tensor h = is_2d ? h0_ : Tensor::zeros(Bi, H);
        const Tensor b_T = b_.transpose(); // (1, 3H)

        Tensor all_out = is_2d ? Tensor::zeros(static_cast<nn::Index>(T_seq), H)
                               : Tensor::zeros(Bi, static_cast<nn::Index>(T_seq), H);

        for (int t = 0; t < T_seq; ++t)
        {
            const Tensor x_t = seq_3d.slice_time(static_cast<nn::Index>(t)); // (B, D)

            const Tensor pre_x = x_t.matmul_transposed(W_).add_row_broadcast(b_T); // (B, 3H) + bias
            const Tensor pre_h = h.matmul_transposed(U_);                          // (B, 3H)

            const Tensor ar = pre_x.block(0, 0 * H, Bi, H).add(pre_h.block(0, 0 * H, Bi, H));
            const Tensor az = pre_x.block(0, 1 * H, Bi, H).add(pre_h.block(0, 1 * H, Bi, H));
            const Tensor uh_n = pre_h.block(0, 2 * H, Bi, H);

            const Tensor r = nn::activations::sigmoid_tensor(ar, exact_activations);
            const Tensor z = nn::activations::sigmoid_tensor(az, exact_activations);
            const Tensor an = pre_x.block(0, 2 * H, Bi, H).add(r * uh_n);
            const Tensor n = nn::activations::tanh_tensor(an, exact_activations);

            const Tensor ones = Tensor::ones(Bi, H);
            const Tensor h_new = ((ones - z) * n).add(z * h);

            if (requires_grad) cache_.push_back({x_t, h, r, z, n, uh_n});

            if (is_2d)
                all_out.setBlock(static_cast<nn::Index>(t), 0, h_new);
            else
                all_out.set_time_slice(static_cast<nn::Index>(t), h_new);

            h = h_new;
        }

        if (is_2d) h0_ = h;
        return all_out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (cache_.empty())
            throw std::runtime_error(
                "GRULayerImpl::backward called before forward(requires_grad=true)");

        const auto& shape = grad_output.get_shape();
        const int B = (shape.size() == 3) ? static_cast<int>(shape[0]) : 1;
        const Tensor go_3d =
            (shape.size() == 2) ? grad_output.reshape({1, shape[0], shape[1]}) : grad_output;

        auto [dW, dU, db, dx_3d] = bptt_pure(go_3d, B);

        W_.set_grad(dW);
        U_.set_grad(dU);
        b_.set_grad(db);
        dW_ = dW;
        dU_ = dU;
        db_ = db;

        if (shape.size() == 2)
        {
            Tensor dx2d = Tensor::zeros(
                static_cast<nn::Index>(shape[0]), static_cast<nn::Index>(input_size_));
            for (int t = 0; t < static_cast<int>(shape[0]); ++t)
                for (int d = 0; d < input_size_; ++d) dx2d.at(t, d) = dx_3d.at(0, t, d);
            return dx2d;
        }
        return dx_3d;
    }

    void reset_state() override
    {
        h0_.set_zero();
        cache_.clear();
    }

    auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        return {{"W", W_}, {"U", U_}, {"b", b_}};
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        if (auto it = sd.find("W"); it != sd.end()) W_ = it->second;
        if (auto it = sd.find("U"); it != sd.end()) U_ = it->second;
        if (auto it = sd.find("b"); it != sd.end()) b_ = it->second;
    }

   private:
    // Derivative of the activation from its cached OUTPUT y (backward has no
    // pre-activation). Exact: sigma' = y(1-y), tanh' = 1-y^2. Fast variants use
    // the closed forms verified in LSTMLayer.hpp.
    auto sigmoid_grad_from_output(const Tensor& y) const -> Tensor
    {
        const Tensor ones = Tensor::ones(y.rows(), y.cols());
        if (exact_activations) return y * (ones - y);
        Tensor d(y.rows(), y.cols());
        for (nn::Index i = 0; i < y.rows(); ++i)
            for (nn::Index j = 0; j < y.cols(); ++j)
            {
                const float s = 1.0F - (2.0F * std::fabs(y.at(i, j) - 0.5F));
                d.at(i, j) = (s * s) * 0.5F;
            }
        return d;
    }
    auto tanh_grad_from_output(const Tensor& y) const -> Tensor
    {
        const Tensor ones = Tensor::ones(y.rows(), y.cols());
        if (exact_activations) return ones - (y * y);
        Tensor d(y.rows(), y.cols());
        for (nn::Index i = 0; i < y.rows(); ++i)
            for (nn::Index j = 0; j < y.cols(); ++j)
            {
                const float t = 1.0F - std::fabs(y.at(i, j));
                d.at(i, j) = t * t;
            }
        return d;
    }

    auto bptt_pure(const Tensor& grad_output, int B) -> std::tuple<Tensor, Tensor, Tensor, Tensor>
    {
        const int T = static_cast<int>(cache_.size());
        const nn::Index H = static_cast<nn::Index>(hidden_size_);
        const nn::Index Bi = static_cast<nn::Index>(B);

        Tensor dW = Tensor::zeros(dW_.rows(), dW_.cols());
        Tensor dU = Tensor::zeros(dU_.rows(), dU_.cols());
        Tensor db = Tensor::zeros(db_.rows(), db_.cols());
        Tensor dx_all =
            Tensor::zeros(Bi, static_cast<nn::Index>(T), static_cast<nn::Index>(input_size_));

        Tensor dh_next = Tensor::zeros(Bi, H);

        for (int t = T - 1; t >= 0; --t)
        {
            const auto& step = cache_[static_cast<std::size_t>(t)];

            Tensor dh = grad_output.slice_time(static_cast<nn::Index>(t)).add(dh_next); // (B, H)

            // H_t = (1 - z) ⊙ n + z ⊙ h_prev
            Tensor dn = dh * (Tensor::ones(Bi, H) - step.z);
            Tensor dz = dh * (step.h_prev - step.n);
            Tensor dh_prev = dh * step.z; // direct path; the recurrent paths add below

            // n = tanh(an),  an = pre_x_n + b_n + r ⊙ uh_n
            Tensor dan = dn * tanh_grad_from_output(step.n);
            Tensor dr = dan * step.uh_n; // into reset gate
            Tensor duh_n = dan * step.r; // into pre_h[:, 2H:3H]

            // r = sigma(ar), z = sigma(az)
            Tensor dar = dr * sigmoid_grad_from_output(step.r);
            Tensor daz = dz * sigmoid_grad_from_output(step.z);

            // Input-side pre-activation grads: [dar | daz | dan]
            Tensor dpre_x(Bi, 3 * H);
            dpre_x.setBlock(0, 0 * H, dar);
            dpre_x.setBlock(0, 1 * H, daz);
            dpre_x.setBlock(0, 2 * H, dan);

            // Hidden-side pre-activation grads: [dar | daz | duh_n]
            Tensor dpre_h(Bi, 3 * H);
            dpre_h.setBlock(0, 0 * H, dar);
            dpre_h.setBlock(0, 1 * H, daz);
            dpre_h.setBlock(0, 2 * H, duh_n);

            // Bias appears only on the input side (an, ar, az) → same layout as dpre_x.
            const Tensor dpre_x_T = dpre_x.transpose();   // (3H, B)
            const Tensor dpre_h_T = dpre_h.transpose();   // (3H, B)
            dW.add_inplace(dpre_x_T.matmul(step.x));      // (3H, D)
            dU.add_inplace(dpre_h_T.matmul(step.h_prev)); // (3H, H)
            db.add_inplace(dpre_x_T.rowwise_sum());       // (3H, 1)

            dx_all.set_time_slice(static_cast<nn::Index>(t), dpre_x.matmul(W_)); // (B, D)
            dh_next = dpre_h.matmul(U_).add(dh_prev);                            // (B, H)
        }

        return {dW, dU, db, dx_all};
    }
};

namespace nn::models::gru
{
using GRULayer = GRULayerImpl<nn::Backend>;
} // namespace nn::models::gru

#endif // NN_LAYERS_GRU_GRULAYER_HPP
