/**
 * @file include/nn/utility/Transforms.hpp
 * @brief PyTorch-inspired input transform pipeline for audio and EEG data.
 *
 * Provides a composable, backend-agnostic transform pipeline modelled after
 * torchvision.transforms.Compose.  All transforms operate on nn::Tensor
 * (rows = batch samples, cols = features) and return a new tensor of the
 * same shape.
 *
 * Included transforms:
 *  - Compose                — sequential chain of transforms.
 *  - AudioMeanStdNormalize  — column-wise mean-variance normalization fitted
 *                             on training data (Simonyan & Zisserman, 2014).
 *  - EEGWindowZScore        — per-window (per-row) z-score (Lotte et al., 2018).
 *  - FusedModalityTransform — applies independent transforms to disjoint
 *                             EEG and audio column ranges within a fused tensor.
 *
 * References
 * ----------
 * Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for
 *   large-scale image recognition. arXiv:1409.1556.
 *
 * Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M.,
 *   Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms
 *   for EEG-based brain-computer interfaces: a 10 year update.
 *   Journal of Neural Engineering, 15(3), 031005.
 *   https://doi.org/10.1088/1741-2552/aab2f2
 *
 * Hardware & performance notes:
 *  - All operations delegate to nn::Tensor, which dispatches to the configured
 *    backend (Eigen CPU or OpenCL GPU).
 *  - AudioMeanStdNormalize accumulates statistics via colwise sums; fitting is
 *    O(N * D) with one pass over training batches.
 *  - EEGWindowZScore is stateless; per-row statistics are computed on the fly.
 *  - FusedModalityTransform performs a shallow copy of the input tensor and
 *    applies setBlock for each modality block.
 */
#ifndef NN_UTILITY_TRANSFORMS_HPP
#define NN_UTILITY_TRANSFORMS_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "nn/tensor/Tensor.hpp"

namespace nn::transforms
{

// ─────────────────────────────────────────────────────────────────────────────
// ITransform — abstract base
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Abstract base for all input transforms.
 *
 * Contract:
 *  - operator() is pure (no side-effects after construction/fitting).
 *  - Input and output tensors share shape: (batch_size, n_features).
 */
class ITransform
{
   public:
    virtual ~ITransform() = default;

    /**
     * Apply the transform to input tensor x.
     * @param x Input tensor  (rows = samples, cols = features).
     * @return  New tensor of the same shape as x.
     */
    virtual auto operator()(const nn::Tensor& x) const -> nn::Tensor = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Compose — sequential pipeline
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Applies a sequence of ITransform instances in insertion order.
 * Equivalent to torchvision.transforms.Compose.
 *
 * Example:
 *   Compose({eeg_zscore, audio_norm})
 *   applies eeg_zscore first, then audio_norm.
 */
class Compose final : public ITransform
{
    std::vector<std::shared_ptr<ITransform>> steps_; ///< Ordered transform steps.

   public:
    /**
     * @param steps Ordered list of transforms to apply.  Must not be empty.
     */
    explicit Compose(std::vector<std::shared_ptr<ITransform>> steps) : steps_(std::move(steps)) {}

    /**
     * Reduces over all steps in insertion order via std::accumulate.
     * @param x Input tensor.
     * @return Tensor after every step has been applied.
     */
    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        return std::accumulate(steps_.cbegin(),
            steps_.cend(),
            x,
            [](const nn::Tensor& acc, const std::shared_ptr<ITransform>& t) { return (*t)(acc); });
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// AudioMeanStdNormalize — fitting-based column-wise mean-variance normalizer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Column-wise mean-variance normalization for audio features.
 *
 * Computes per-feature (column) mean μ_j and standard deviation σ_j from
 * accumulated training batches, then applies:
 *
 *   x'[i, j] = (x[i, j] − μ_j) / (σ_j + ε)
 *
 * Two-stage usage:
 *   1. Call accumulate(batch) for every training batch.
 *   2. Call finalize() once to compute μ and σ.
 *   3. operator()(batch) applies the fitted normalizer.
 *
 * Reference:
 *   Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks
 *   for large-scale image recognition. arXiv:1409.1556.
 */
class AudioMeanStdNormalize final : public ITransform
{
    nn::Tensor mean_;     ///< Column means  (1, n_features); valid after finalize().
    nn::Tensor std_;      ///< Column std devs (1, n_features); valid after finalize().
    bool fitted_ = false; ///< True after finalize() has been called.
    float eps_;           ///< Stability floor added inside sqrt.

    // Accumulation buffers (released after finalize).
    nn::Tensor sum_col_;     ///< Running colwise sum   (1, n_features).
    nn::Tensor sum_sq_col_;  ///< Running colwise sum²  (1, n_features).
    float n_samples_ = 0.0F; ///< Total number of accumulated samples.

   public:
    /**
     * @param eps Numerical stability constant added inside the square root.
     *            Prevents division-by-zero on constant features.
     */
    explicit AudioMeanStdNormalize(float eps = 1e-6F) : eps_(eps) {}

    /**
     * Accumulate column statistics from one training batch.
     * Must be called before finalize().
     * @param batch Training batch (rows = samples, cols = audio features).
     */
    void accumulate(const nn::Tensor& batch)
    {
        if (batch.rows() == 0 || batch.cols() == 0)
        {
            return;
        }

        const auto col_sum = batch.sum_cols();         // (1, cols)
        const auto sq_sum = batch.square().sum_cols(); // (1, cols)

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

    /**
     * Finalize: compute column means and standard deviations from accumulated sums.
     * Must be called exactly once after all accumulate() calls, before operator().
     * @throws std::runtime_error if called with zero accumulated samples.
     */
    void finalize()
    {
        if (n_samples_ <= 0.0F)
        {
            throw std::runtime_error(
                "AudioMeanStdNormalize::finalize() called with zero accumulated samples");
        }

        mean_ = sum_col_;
        mean_.divide_scalar_inplace(n_samples_);

        // Per-column: var_j = E[x²] − E[x]²; std_j = sqrt(max(var_j, 0) + ε).
        std_ = nn::Tensor(1, mean_.cols());
        for (nn::Index j = 0; j < mean_.cols(); ++j)
        {
            const float mean_j = mean_.at(0, j);
            const float e_sq = sum_sq_col_.at(0, j) / n_samples_;
            const float var_j = e_sq - mean_j * mean_j;
            std_.at(0, j) = std::sqrt(std::max(0.0F, var_j) + eps_);
        }

        // Release accumulation buffers — no longer needed.
        sum_col_ = nn::Tensor{};
        sum_sq_col_ = nn::Tensor{};
        fitted_ = true;
    }

    /** @return true iff finalize() has been successfully called. */
    [[nodiscard]] auto is_fitted() const noexcept -> bool
    {
        return fitted_;
    }

    /**
     * Apply fitted column-wise normalizer.
     * @param x Input tensor (rows = samples, cols = audio features).
     * @return Normalized tensor of the same shape.
     * @throws std::runtime_error    if finalize() has not been called.
     * @throws std::invalid_argument if x.cols() ≠ training feature count.
     */
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
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// EEGWindowZScore — stateless per-window (per-row) z-score normalizer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-window z-score normalization for EEG data.
 *
 * For each sample row (one EEG time window), computes the window mean μ_i
 * and standard deviation σ_i, then normalizes every feature within that row:
 *
 *   x'[i, j] = (x[i, j] − μ_i) / (σ_i + ε)
 *
 * Stateless: no fitting pass is required.  Statistics are computed locally
 * within each window at transform time, ensuring zero data leakage.
 *
 * Reference:
 *   Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M.,
 *   Rakotomamonjy, A., & Yger, F. (2018). A review of classification
 *   algorithms for EEG-based brain-computer interfaces: a 10 year update.
 *   Journal of Neural Engineering, 15(3), 031005.
 *   https://doi.org/10.1088/1741-2552/aab2f2
 */
class EEGWindowZScore final : public ITransform
{
    float eps_; ///< Stability floor preventing division-by-zero on flat windows.

   public:
    /**
     * @param eps Stability constant added inside the per-row standard deviation.
     */
    explicit EEGWindowZScore(float eps = 1e-6F) : eps_(eps) {}

    /**
     * Apply window-based z-score.  Each row is normalized independently.
     * @param x Input tensor (rows = windows, cols = EEG features).
     * @return Normalized tensor of the same shape.
     */
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
            // One-pass Welford variant: accumulate sum and sum-of-squares.
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

// ─────────────────────────────────────────────────────────────────────────────
// FusedModalityTransform — modality-aware column-range dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Applies independent transforms to the EEG and audio column ranges within
 * a fused (EEG ‖ audio) batch tensor.
 *
 * Expected column layout (rows = samples):
 *   cols [0,          eeg_cols_)                 → EEG features  → eeg_transform_
 *   cols [eeg_cols_,  eeg_cols_ + audio_cols_)   → audio features → audio_transform_
 *
 * Either transform pointer may be null, in which case the corresponding
 * column range is passed through unmodified.
 */
class FusedModalityTransform final : public ITransform
{
    nn::Index eeg_cols_;   ///< Number of EEG feature columns (starting at 0).
    nn::Index audio_cols_; ///< Number of audio feature columns (after EEG).

    std::shared_ptr<ITransform> eeg_transform_;   ///< Transform for EEG block; may be null.
    std::shared_ptr<ITransform> audio_transform_; ///< Transform for audio block; may be null.

   public:
    /**
     * @param eeg_cols         Number of EEG columns  (cols 0 .. eeg_cols-1).
     * @param audio_cols       Number of audio columns (cols eeg_cols .. eeg_cols+audio_cols-1).
     * @param eeg_transform    Applied to the EEG block; pass nullptr to leave unchanged.
     * @param audio_transform  Applied to the audio block; pass nullptr to leave unchanged.
     */
    FusedModalityTransform(nn::Index eeg_cols,
        nn::Index audio_cols,
        std::shared_ptr<ITransform> eeg_transform,
        std::shared_ptr<ITransform> audio_transform)
        : eeg_cols_(eeg_cols),
          audio_cols_(audio_cols),
          eeg_transform_(std::move(eeg_transform)),
          audio_transform_(std::move(audio_transform))
    {
    }

    /**
     * Apply modality-specific transforms.
     * EEG and audio blocks are processed independently and recombined.
     * @param x Fused input tensor (rows = samples, cols ≥ eeg_cols + audio_cols).
     * @return New tensor of the same shape with each modality block normalized.
     */
    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        auto out = x; // Shallow copy; modified regions replaced via setBlock.

        if (eeg_transform_ && eeg_cols_ > 0)
        {
            const auto eeg_block = x.block(0, 0, x.rows(), eeg_cols_);
            const auto eeg_normalized = (*eeg_transform_)(eeg_block);
            out.setBlock(0, 0, eeg_normalized);
        }

        if (audio_transform_ && audio_cols_ > 0)
        {
            const auto audio_block = x.block(0, eeg_cols_, x.rows(), audio_cols_);
            const auto audio_normalized = (*audio_transform_)(audio_block);
            out.setBlock(0, eeg_cols_, audio_normalized);
        }

        return out;
    }
};

} // namespace nn::transforms

#endif // NN_UTILITY_TRANSFORMS_HPP
