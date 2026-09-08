#ifndef NN_LAYERS_NORMALIZATION_LAYERNORM_HPP
#define NN_LAYERS_NORMALIZATION_LAYERNORM_HPP

/**
 * @file include/layers/normalization/LayerNorm.hpp
 * @brief Layer normalization over the feature (column) dimension of a 2-D tensor.
 *
 * For a row x in R^D:
 *   mu    = mean_j x_j
 *   var   = mean_j (x_j - mu)^2
 *   xhat  = (x - mu) / sqrt(var + eps)
 *   y     = gamma ⊙ xhat + beta
 *
 * gamma, beta are learned length-D vectors (stored (1, D)); gamma init 1, beta 0.
 * Input/output shape (N, D); N rows are normalized independently. Validated by a
 * finite-difference gradient check (layernorm_gtest.cpp).
 *
 * Reference: Ba, Kiros & Hinton, "Layer Normalization", arXiv:1607.06450 (2016).
 */

#include <cmath>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

template <typename Backend>
class LayerNormImpl : public Module<Backend>
{
   public:
    using Tensor = nn::TensorImpl<Backend>;

    int normalized_size_;
    float eps_;

    Tensor gamma; // (1, D)
    Tensor beta;  // (1, D)

    Tensor xhat_cache_;    // (N, D)
    Tensor inv_std_cache_; // (N, 1)
    bool requires_grad_ = false;
    bool forward_cached_ = false;

    std::vector<Tensor*> param_ptrs_;

    explicit LayerNormImpl(int normalized_size, float eps = 1e-5f)
        : normalized_size_(normalized_size),
          eps_(eps),
          gamma(1, normalized_size),
          beta(1, normalized_size)
    {
        for (nn::Index j = 0; j < static_cast<nn::Index>(normalized_size_); ++j)
        {
            gamma.at(0, j) = 1.0f;
            beta.at(0, j) = 0.0f;
        }
        param_ptrs_ = {&gamma, &beta};
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        requires_grad_ = requires_grad;
        if (input.get_shape().size() != 2 || static_cast<int>(input.cols()) != normalized_size_)
            throw std::invalid_argument(
                "LayerNormImpl::forward: expected (N, " + std::to_string(normalized_size_) + ")");

        const nn::Index N = input.rows();
        const nn::Index D = input.cols();
        const float df = static_cast<float>(D);

        Tensor out(N, D);
        Tensor xhat(N, D);
        Tensor inv_std(N, 1);

        for (nn::Index i = 0; i < N; ++i)
        {
            float mean = 0.0f;
            for (nn::Index j = 0; j < D; ++j) mean += input.at(i, j);
            mean /= df;

            float var = 0.0f;
            for (nn::Index j = 0; j < D; ++j)
            {
                const float c = input.at(i, j) - mean;
                var += c * c;
            }
            var /= df;

            const float is = 1.0f / std::sqrt(var + eps_);
            inv_std.at(i, 0) = is;
            for (nn::Index j = 0; j < D; ++j)
            {
                const float xh = (input.at(i, j) - mean) * is;
                xhat.at(i, j) = xh;
                out.at(i, j) = gamma.at(0, j) * xh + beta.at(0, j);
            }
        }

        if (requires_grad)
        {
            xhat_cache_ = xhat;
            inv_std_cache_ = inv_std;
            forward_cached_ = true;
        }
        return out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        if (!forward_cached_)
            throw std::runtime_error(
                "LayerNormImpl::backward called before forward(requires_grad=true)");

        const nn::Index N = grad_output.rows();
        const nn::Index D = grad_output.cols();
        const float df = static_cast<float>(D);

        Tensor dx(N, D);
        Tensor dgamma(1, D);
        Tensor dbeta(1, D);
        for (nn::Index j = 0; j < D; ++j)
        {
            dgamma.at(0, j) = 0.0f;
            dbeta.at(0, j) = 0.0f;
        }

        for (nn::Index i = 0; i < N; ++i)
        {
            const float is = inv_std_cache_.at(i, 0);

            float sum_dxhat = 0.0f;
            float sum_dxhat_xhat = 0.0f;
            for (nn::Index j = 0; j < D; ++j)
            {
                const float go = grad_output.at(i, j);
                const float xh = xhat_cache_.at(i, j);
                const float dxhat = go * gamma.at(0, j);
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * xh;
                dgamma.at(0, j) += go * xh;
                dbeta.at(0, j) += go;
            }

            for (nn::Index j = 0; j < D; ++j)
            {
                const float xh = xhat_cache_.at(i, j);
                const float dxhat = grad_output.at(i, j) * gamma.at(0, j);
                dx.at(i, j) = (is / df) * (df * dxhat - sum_dxhat - xh * sum_dxhat_xhat);
            }
        }

        gamma.set_grad(dgamma);
        beta.set_grad(dbeta);
        return dx;
    }

    void reset_state() override
    {
        xhat_cache_ = Tensor();
        inv_std_cache_ = Tensor();
        forward_cached_ = false;
    }

    auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        return {{"gamma", gamma}, {"beta", beta}};
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        if (auto it = sd.find("gamma"); it != sd.end()) gamma = it->second;
        if (auto it = sd.find("beta"); it != sd.end()) beta = it->second;
    }
};

namespace nn::layers
{
using LayerNorm = LayerNormImpl<nn::Backend>;
} // namespace nn::layers

#endif // NN_LAYERS_NORMALIZATION_LAYERNORM_HPP
