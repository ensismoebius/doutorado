#ifndef NN_UTILITY_GRADCLIP_HPP
#define NN_UTILITY_GRADCLIP_HPP

#include <cmath>
#include <span>

#include "tensor/Tensor.hpp"

namespace nn::utils {

// Global-norm gradient clipping (PyTorch-style).
// Computes ||g|| over all parameters; if it exceeds max_norm, scales every
// gradient tensor by max_norm/||g||. No-op when global norm <= max_norm.
template <typename Tensor>
inline void clip_grad_norm(std::span<Tensor*> params, float max_norm)
{
    float total_sq = 0.0f;
    for (Tensor* p : params)
    {
        float n = p->grad().norm();
        total_sq += n * n;
    }
    const float global_norm = std::sqrt(total_sq);
    if (global_norm > max_norm && global_norm > 0.0f)
    {
        const float scale = max_norm / global_norm;
        for (Tensor* p : params)
        {
            Tensor g = p->grad();
            g.multiply_scalar_inplace(scale);
            p->set_grad(g);
        }
    }
}

} // namespace nn::utils

#endif // NN_UTILITY_GRADCLIP_HPP
