#ifndef NN_LAYERS_ATTENTION_TRANSFORMERENCODERBLOCK_HPP
#define NN_LAYERS_ATTENTION_TRANSFORMERENCODERBLOCK_HPP

/**
 * @file include/layers/attention/TransformerEncoderBlock.hpp
 * @brief One Transformer encoder block, post-norm (Vaswani et al., 2017; the default) or
 *        pre-norm (Xiong et al., ICML 2020).
 *
 *   post-norm (norm_first = false)            pre-norm (norm_first = true)
 *   a    = MultiHeadAttention(x)              a    = MultiHeadAttention(LayerNorm1(x))
 *   n1   = LayerNorm1(x + a)                  h    = x + a
 *   f    = FFN(n1)                            f    = FFN(LayerNorm2(h))
 *   out  = LayerNorm2(n1 + f)                 out  = h + f
 *   FFN(v) = W2 * ReLU(W1 * v + b1) + b2      (position-wise)
 *
 * Why both: Xiong et al. show that at initialisation the post-norm block's gradients near the
 * output are large, so Adam at a fixed learning rate needs a warm-up, while the pre-norm block
 * keeps an identity path from input to output and trains without one. The meeting01
 * Transformer-AE (post-norm, Adam 1e-3, no warm-up) never left the mean-predictor plateau in
 * the 55 trainings of GridUnesp job 5315335; the pre-norm variant is how that is tested. A
 * pre-norm stack is normally closed by a final LayerNorm, which the model adds
 * (TransformerAutoencoder), not the block.
 *
 * All sub-modules are individually finite-difference gradient-checked; this block
 * adds a composed check of each variant (transformer_encoder_block_gtest.cpp). Self-attention
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
    bool norm_first_; // pre-norm (Xiong et al. 2020) instead of post-norm

    std::unique_ptr<MultiHeadAttentionImpl<Backend>> mha_;
    std::unique_ptr<LayerNormImpl<Backend>> ln1_;
    std::unique_ptr<Linear> ff1_;
    std::unique_ptr<ReLUImpl<Backend>> relu_;
    std::unique_ptr<Linear> ff2_;
    std::unique_ptr<LayerNormImpl<Backend>> ln2_;

    bool forward_cached_ = false;
    std::vector<Tensor*> param_ptrs_;

    TransformerEncoderBlockImpl(
        int d_model, int n_heads, int d_ff, unsigned seed = 0u, bool norm_first = false)
        : d_model_(d_model), d_ff_(d_ff), norm_first_(norm_first)
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

        const auto ffn = [&](const Tensor& v)
        {
            return ff2_->forward(
                relu_->forward(ff1_->forward(v, requires_grad), requires_grad), requires_grad);
        };
        Tensor out;
        if (norm_first_)
        {
            const Tensor h = x.add(mha_->forward(ln1_->forward(x, requires_grad), requires_grad));
            out = h.add(ffn(ln2_->forward(h, requires_grad)));
        }
        else
        {
            const Tensor a = mha_->forward(x, requires_grad);
            const Tensor n1 = ln1_->forward(x.add(a), requires_grad);
            out = ln2_->forward(n1.add(ffn(n1)), requires_grad);
        }

        if (requires_grad) forward_cached_ = true;
        return out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (!forward_cached_)
            throw std::runtime_error(
                "TransformerEncoderBlockImpl::backward called before forward(requires_grad=true)");

        if (norm_first_)
        {
            // out = h + FFN(LayerNorm2(h)): the residual and the FFN path both reach h.
            Tensor d_u = ff2_->backward(grad_output);
            d_u = relu_->backward(d_u);
            const Tensor d_h = grad_output.add(ln2_->backward(ff1_->backward(d_u)));
            // h = x + MHA(LayerNorm1(x))
            return d_h.add(ln1_->backward(mha_->backward(d_h)));
        }

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
