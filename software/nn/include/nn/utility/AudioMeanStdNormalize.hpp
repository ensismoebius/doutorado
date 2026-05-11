#ifndef AUDIO_MEAN_STD_NORMALIZE_HPP
#define AUDIO_MEAN_STD_NORMALIZE_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#include "nn/utility/ITransform.hpp"

namespace nn::transforms
{

class AudioMeanStdNormalize final : public ITransform
{
    nn::Tensor mean_;
    nn::Tensor std_;
    bool fitted_ = false;
    float eps_;

    nn::Tensor sum_col_;
    nn::Tensor sum_sq_col_;
    float n_samples_ = 0.0F;

   public:
    explicit AudioMeanStdNormalize(float eps = 1e-6F) : eps_(eps) {}

    void accumulate(const nn::Tensor& batch)
    {
        if (batch.rows() == 0 || batch.cols() == 0)
        {
            return;
        }

        const auto col_sum = batch.sum_cols();
        const auto sq_sum = batch.square().sum_cols();

        if (n_samples_ == 0.0F)
        {
            sum_col_ = col_sum;
            sum_sq_col_ = sq_sum;
        }
        else
        {
            sum_col_.add_inplace(col_sum);
            sum_sq_col_.add_inplace(sq_sum);
        }
        n_samples_ += static_cast<float>(batch.rows());
    }

    void finalize()
    {
        if (n_samples_ <= 0.0F)
        {
            throw std::runtime_error(
                "AudioMeanStdNormalize::finalize() called with zero accumulated samples");
        }

        mean_ = sum_col_;
        mean_.divide_scalar_inplace(n_samples_);

        std_ = nn::Tensor(1, mean_.cols());
        for (nn::Index j = 0; j < mean_.cols(); ++j)
        {
            const float mean_j = mean_.at(0, j);
            const float e_sq = sum_sq_col_.at(0, j) / n_samples_;
            const float var_j = e_sq - mean_j * mean_j;
            std_.at(0, j) = std::sqrt(std::max(0.0F, var_j) + eps_);
        }

        sum_col_ = nn::Tensor{};
        sum_sq_col_ = nn::Tensor{};
        fitted_ = true;
    }

    [[nodiscard]] auto is_fitted() const noexcept -> bool
    {
        return fitted_;
    }

    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        if (!fitted_)
        {
            throw std::runtime_error(
                "AudioMeanStdNormalize: call finalize() before applying the transform");
        }
        if (x.cols() != mean_.cols())
        {
            throw std::invalid_argument(
                "AudioMeanStdNormalize: feature dimension mismatch at inference time");
        }

        nn::Tensor out(x.rows(), x.cols());
        for (nn::Index i = 0; i < x.rows(); ++i)
        {
            for (nn::Index j = 0; j < x.cols(); ++j)
            {
                out.at(i, j) = (x.at(i, j) - mean_.at(0, j)) / std_.at(0, j);
            }
        }
        return out;
    } // LCOV_EXCL_LINE
};

} // namespace nn::transforms

#endif // AUDIO_MEAN_STD_NORMALIZE_HPP