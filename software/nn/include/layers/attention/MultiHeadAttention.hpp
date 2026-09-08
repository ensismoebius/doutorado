#ifndef NN_LAYERS_ATTENTION_MULTIHEADATTENTION_HPP
#define NN_LAYERS_ATTENTION_MULTIHEADATTENTION_HPP

/**
 * @file include/layers/attention/MultiHeadAttention.hpp
 * @brief Multi-head self-attention with full backward.
 *
 * Self-attention only (Q, K, V all come from the same input X), which is all the
 * encoder-only Transformer autoencoder needs. For input X in R^{T x d_model}:
 *
 *   Q = X Wq^T + bq,  K = X Wk^T + bk,  V = X Wv^T + bv          // (T, d_model)
 *   per head h (d_k = d_model / n_heads):
 *     S_h = (Q_h K_h^T) / sqrt(d_k)                              // (T, T)
 *     A_h = softmax_rows(S_h)
 *     O_h = A_h V_h                                              // (T, d_k)
 *   O = concat_h O_h                                             // (T, d_model)
 *   out = O Wo^T + bo
 *
 * The Q/K/V/O projections are LinearImpl sub-modules, so their parameter
 * gradients come from LinearImpl::backward. The attention core (matmuls +
 * row-softmax) is differentiated explicitly here and validated by a
 * finite-difference gradient check (multi_head_attention_gtest.cpp).
 *
 * Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
 */

#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/base/Module.hpp"
#include "layers/dense/Linear.hpp"
#include "tensor/Tensor.hpp"

template <typename Backend>
struct MHAHeadCacheT
{
    using Tensor = nn::TensorImpl<Backend>;
    Tensor qh; // (T, d_k)
    Tensor kh; // (T, d_k)
    Tensor vh; // (T, d_k)
    Tensor a;  // (T, T) row-softmax weights
};

template <typename Backend>
class MultiHeadAttentionImpl : public Module<Backend>
{
   public:
    using Tensor = nn::TensorImpl<Backend>;
    using Linear = LinearImpl<Backend>;

    int d_model_;
    int n_heads_;
    int d_k_;
    float scale_;

    std::unique_ptr<Linear> lin_q_;
    std::unique_ptr<Linear> lin_k_;
    std::unique_ptr<Linear> lin_v_;
    std::unique_ptr<Linear> lin_o_;

    std::vector<MHAHeadCacheT<Backend>> head_cache_;
    bool forward_cached_ = false;

    std::vector<Tensor*> param_ptrs_;

    MultiHeadAttentionImpl(int d_model, int n_heads, unsigned seed = 0u)
        : d_model_(d_model),
          n_heads_(n_heads),
          d_k_((n_heads > 0 && d_model % n_heads == 0) ? d_model / n_heads : 0),
          scale_(d_k_ > 0 ? 1.0f / std::sqrt(static_cast<float>(d_k_)) : 1.0f)
    {
        if (n_heads <= 0 || d_model <= 0 || d_model % n_heads != 0)
            throw std::invalid_argument(
                "MultiHeadAttentionImpl: need d_model > 0, n_heads > 0, and n_heads | d_model");

        lin_q_ = std::make_unique<Linear>(d_model, d_model);
        lin_k_ = std::make_unique<Linear>(d_model, d_model);
        lin_v_ = std::make_unique<Linear>(d_model, d_model);
        lin_o_ = std::make_unique<Linear>(d_model, d_model);

        auto init = [&](Linear& lin, unsigned s)
        {
            std::mt19937 rng(1234u + seed + s);
            std::normal_distribution<float> dist(0.0f, 0.05f);
            for (nn::Index k = 0; k < static_cast<nn::Index>(lin.weight.size()); ++k)
                lin.weight.at(k) = dist(rng);
            lin.bias.set_zero();
        };
        init(*lin_q_, 0u);
        init(*lin_k_, 1u);
        init(*lin_v_, 2u);
        init(*lin_o_, 3u);

        rebuild_param_ptrs();
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (input.get_shape().size() != 2 || static_cast<int>(input.cols()) != d_model_)
            throw std::invalid_argument("MultiHeadAttentionImpl::forward: expected (T, d_model)");

        const nn::Index T = input.rows();
        const nn::Index dk = static_cast<nn::Index>(d_k_);

        const Tensor Q = lin_q_->forward(input, requires_grad);
        const Tensor K = lin_k_->forward(input, requires_grad);
        const Tensor V = lin_v_->forward(input, requires_grad);

        Tensor O(T, static_cast<nn::Index>(d_model_));
        if (requires_grad) head_cache_.clear();

        for (int h = 0; h < n_heads_; ++h)
        {
            const nn::Index c0 = static_cast<nn::Index>(h) * dk;
            const Tensor Qh = Q.block(0, c0, T, dk);
            const Tensor Kh = K.block(0, c0, T, dk);
            const Tensor Vh = V.block(0, c0, T, dk);

            Tensor S = Qh.matmul(Kh.transpose()) * scale_; // (T, T)
            const Tensor A = softmax_rows(S);
            const Tensor Oh = A.matmul(Vh); // (T, d_k)
            O.setBlock(0, c0, Oh);

            if (requires_grad) head_cache_.push_back({Qh, Kh, Vh, A});
        }

        const Tensor out = lin_o_->forward(O, requires_grad);
        if (requires_grad) forward_cached_ = true;
        return out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (!forward_cached_)
            throw std::runtime_error(
                "MultiHeadAttentionImpl::backward called before forward(requires_grad=true)");

        const nn::Index T = grad_output.rows();
        const nn::Index dk = static_cast<nn::Index>(d_k_);

        const Tensor dO = lin_o_->backward(grad_output); // (T, d_model)

        Tensor dQ = Tensor::zeros(T, static_cast<nn::Index>(d_model_));
        Tensor dK = Tensor::zeros(T, static_cast<nn::Index>(d_model_));
        Tensor dV = Tensor::zeros(T, static_cast<nn::Index>(d_model_));

        for (int h = 0; h < n_heads_; ++h)
        {
            const nn::Index c0 = static_cast<nn::Index>(h) * dk;
            const auto& c = head_cache_[static_cast<std::size_t>(h)];

            const Tensor dOh = dO.block(0, c0, T, dk);

            // O_h = A_h V_h
            const Tensor dA = dOh.matmul(c.vh.transpose()); // (T, T)
            const Tensor dVh = c.a.transpose().matmul(dOh); // (T, d_k)

            // Row-softmax backward, then the 1/sqrt(d_k) scale.
            Tensor dS = softmax_rows_backward(c.a, dA) * scale_; // (T, T)

            // S_h = Q_h K_h^T
            const Tensor dQh = dS.matmul(c.kh);             // (T, d_k)
            const Tensor dKh = dS.transpose().matmul(c.qh); // (T, d_k)

            dQ.setBlock(0, c0, dQh);
            dK.setBlock(0, c0, dKh);
            dV.setBlock(0, c0, dVh);
        }

        Tensor dX = lin_q_->backward(dQ);
        dX = dX.add(lin_k_->backward(dK));
        dX = dX.add(lin_v_->backward(dV));
        return dX;
    }

    void reset_state() override
    {
        head_cache_.clear();
        forward_cached_ = false;
    }

    auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        std::map<std::string, Tensor> sd;
        auto merge = [&](const std::string& pfx, const std::map<std::string, Tensor>& src)
        {
            for (const auto& [k, v] : src) sd[pfx + k] = v;
        };
        merge("q.", lin_q_->state_dict());
        merge("k.", lin_k_->state_dict());
        merge("v.", lin_v_->state_dict());
        merge("o.", lin_o_->state_dict());
        return sd;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        auto sub = [&](const std::string& pfx)
        {
            std::map<std::string, Tensor> out;
            for (const auto& [k, v] : sd)
                if (k.rfind(pfx, 0) == 0) out[k.substr(pfx.size())] = v;
            return out;
        };
        lin_q_->load_state_dict(sub("q."));
        lin_k_->load_state_dict(sub("k."));
        lin_v_->load_state_dict(sub("v."));
        lin_o_->load_state_dict(sub("o."));
    }

   private:
    void rebuild_param_ptrs()
    {
        param_ptrs_.clear();
        for (Linear* l : {lin_q_.get(), lin_k_.get(), lin_v_.get(), lin_o_.get()})
            for (Tensor* p : l->params()) param_ptrs_.push_back(p);
    }

    // Row-wise softmax with max subtraction for numerical stability.
    static auto softmax_rows(const Tensor& s) -> Tensor
    {
        const nn::Index R = s.rows();
        const nn::Index C = s.cols();
        Tensor out(R, C);
        for (nn::Index i = 0; i < R; ++i)
        {
            float m = s.at(i, 0);
            for (nn::Index j = 1; j < C; ++j) m = std::max(m, s.at(i, j));
            float denom = 0.0f;
            for (nn::Index j = 0; j < C; ++j)
            {
                const float e = std::exp(s.at(i, j) - m);
                out.at(i, j) = e;
                denom += e;
            }
            for (nn::Index j = 0; j < C; ++j) out.at(i, j) /= denom;
        }
        return out;
    }

    // Given the softmax output a and upstream grad da, returns grad wrt the
    // pre-softmax scores: ds[i,:] = a[i,:] ⊙ (da[i,:] - <da[i,:], a[i,:]>).
    static auto softmax_rows_backward(const Tensor& a, const Tensor& da) -> Tensor
    {
        const nn::Index R = a.rows();
        const nn::Index C = a.cols();
        Tensor ds(R, C);
        for (nn::Index i = 0; i < R; ++i)
        {
            float dot = 0.0f;
            for (nn::Index j = 0; j < C; ++j) dot += da.at(i, j) * a.at(i, j);
            for (nn::Index j = 0; j < C; ++j) ds.at(i, j) = a.at(i, j) * (da.at(i, j) - dot);
        }
        return ds;
    }
};

namespace nn::layers
{
using MultiHeadAttention = MultiHeadAttentionImpl<nn::Backend>;
} // namespace nn::layers

#endif // NN_LAYERS_ATTENTION_MULTIHEADATTENTION_HPP
