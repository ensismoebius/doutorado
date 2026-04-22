#ifndef EEG_WINDOW_ZSCORE_HPP
#define EEG_WINDOW_ZSCORE_HPP

#include <cmath>

#include "nn/utility/ITransform.hpp"

namespace nn::transforms
{

class EEGWindowZScore final : public ITransform
{
    float eps_;

   public:
    explicit EEGWindowZScore(float eps = 1e-6F) : eps_(eps) {}

    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        if (x.rows() == 0 || x.cols() == 0)
        {
            return x;
        }

        const float inv_cols = 1.0F / static_cast<float>(x.cols());
        nn::Tensor out(x.rows(), x.cols());

        for (nn::Index i = 0; i < x.rows(); ++i)
        {
            float sum = 0.0F;
            float sum_sq = 0.0F;
            for (nn::Index j = 0; j < x.cols(); ++j)
            {
                const float v = x.at(i, j);
                sum += v;
                sum_sq += v * v;
            }
            const float mu = sum * inv_cols;
            const float sigma = std::sqrt(std::max(0.0F, sum_sq * inv_cols - mu * mu) + eps_);

            for (nn::Index j = 0; j < x.cols(); ++j)
            {
                out.at(i, j) = (x.at(i, j) - mu) / sigma;
            }
        }
        return out;
    }
};

} // namespace nn::transforms

#endif // EEG_WINDOW_ZSCORE_HPP