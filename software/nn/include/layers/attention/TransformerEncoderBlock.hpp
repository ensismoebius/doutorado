#ifndef NN_LAYERS_ATTENTION_TRANSFORMERENCODERBLOCK_HPP
#define NN_LAYERS_ATTENTION_TRANSFORMERENCODERBLOCK_HPP

/**
 * @file include/layers/attention/TransformerEncoderBlock.hpp
 * @brief One post-norm Transformer encoder block (Vaswani et al., 2017).
 *
 *   a    = MultiHeadAttention(x)
 *   n1   = LayerNorm1(x + a)
 *   f    = W2 * ReLU(W1 * n1 + b1) + b2                 // position-wise FFN
 *   out  = LayerNorm2(n1 + f)
 *
 * All sub-modules are individually finite-difference gradient-checked; this block
 * adds a composed check (transformer_encoder_block_gtest.cpp). Self-attention
 * only — sufficient for the encoder-only / bottlenecked Transformer autoencoder.
 */

#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/activations/ReLU.hpp"
#include "layers/attention/MultiHeadAttention.hpp"
#include "layers/base/Module.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/normalization/LayerNorm.hpp"
#include "tensor/Tensor.hpp"

template <typename Backend>
class TransformerEncoderBlockImpl : public Module<Backend>
{
   public:
    using Tensor = nn::TensorImpl<Backend>;
    using Linear = LinearImpl<Backend>;

    int d_model_;
    int d_ff_;

    std::unique_ptr<MultiHeadAttentionImpl<Backend>> mha_;
    std::unique_ptr<LayerNormImpl<Backend>> ln1_;
    std::unique_ptr<Linear> ff1_;
    std::unique_ptr<ReLUImpl<Backend>> relu_;
    std::unique_ptr<Linear> ff2_;
    std::unique_ptr<LayerNormImpl<Backend>> ln2_;

    bool forward_cached_ = false;
    std::vector<Tensor*> param_ptrs_;

    TransformerEncoderBlockImpl(int d_model, int n_heads, int d_ff, unsigned seed = 0u)
        : d_model_(d_model), d_ff_(d_ff)
    {
        if (d_model <= 0 || d_ff <= 0)
            throw std::invalid_argument("TransformerEncoderBlockImpl: d_model, d_ff must be > 0");

        mha_ = std::make_unique<MultiHeadAttentionImpl<Backend>>(d_model, n_heads, seed);
        ln1_ = std::make_unique<LayerNormImpl<Backend>>(d_model);
        ff1_ = std::make_unique<Linear>(d_model, d_ff);
        relu_ = std::make_unique<ReLUImpl<Backend>>();
        ff2_ = std::make_unique<Linear>(d_ff, d_model);
        ln2_ = std::make_unique<LayerNormImpl<Backend>>(d_model);

        auto init = [&](Linear& lin, unsigned s)
        {
            std::mt19937 rng(9000u + seed + s);
            std::normal_distribution<float> dist(0.0f, 0.05f);
            for (nn::Index k = 0; k < static_cast<nn::Index>(lin.weight.size()); ++k)
                lin.weight.at(k) = dist(rng);
            lin.bias.set_zero();
        };
        init(*ff1_, 0u);
        init(*ff2_, 1u);

        rebuild_param_ptrs();
    }

    auto forward(const Tensor& x, bool requires_grad = true) -> Tensor override
    {
        if (x.get_shape().size() != 2 || static_cast<int>(x.cols()) != d_model_)
            throw std::invalid_argument(
                "TransformerEncoderBlockImpl::forward: expected (T, d_model)");

        const Tensor a = mha_->forward(x, requires_grad);
        const Tensor n1 = ln1_->forward(x.add(a), requires_grad);
        const Tensor f = ff2_->forward(
            relu_->forward(ff1_->forward(n1, requires_grad), requires_grad), requires_grad);
        const Tensor out = ln2_->forward(n1.add(f), requires_grad);

        if (requires_grad) forward_cached_ = true;
        return out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (!forward_cached_)
            throw std::runtime_error(
                "TransformerEncoderBlockImpl::backward called before forward(requires_grad=true)");

        // out = LayerNorm2(n1 + f)
        const Tensor d_res2 = ln2_->backward(grad_output);
        // res2 = n1 + f  → gradient splits equally
        const Tensor d_f = d_res2;
        Tensor d_n1 = d_res2;

        // f = ff2(relu(ff1(n1)))
        Tensor d_h = ff2_->backward(d_f);
        d_h = relu_->backward(d_h);
        d_n1 = d_n1.add(ff1_->backward(d_h));

        // n1 = LayerNorm1(x + a)
        const Tensor d_res1 = ln1_->backward(d_n1);
        // res1 = x + a  → gradient splits
        Tensor d_x = d_res1;
        d_x = d_x.add(mha_->backward(d_res1));
        return d_x;
    }

    void reset_state() override
    {
        mha_->reset_state();
        ln1_->reset_state();
        ln2_->reset_state();
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
        merge("mha.", mha_->state_dict());
        merge("ln1.", ln1_->state_dict());
        merge("ff1.", ff1_->state_dict());
        merge("ff2.", ff2_->state_dict());
        merge("ln2.", ln2_->state_dict());
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
        mha_->load_state_dict(sub("mha."));
        ln1_->load_state_dict(sub("ln1."));
        ff1_->load_state_dict(sub("ff1."));
        ff2_->load_state_dict(sub("ff2."));
        ln2_->load_state_dict(sub("ln2."));
    }

   private:
    void rebuild_param_ptrs()
    {
        param_ptrs_.clear();
        auto take = [&](std::span<Tensor*> ps)
        {
            for (Tensor* p : ps) param_ptrs_.push_back(p);
        };
        take(mha_->params());
        take(ln1_->params());
        take(ff1_->params());
        take(ff2_->params());
        take(ln2_->params());
    }
};

namespace nn::layers
{
using TransformerEncoderBlock = TransformerEncoderBlockImpl<nn::Backend>;
} // namespace nn::layers

#endif // NN_LAYERS_ATTENTION_TRANSFORMERENCODERBLOCK_HPP
