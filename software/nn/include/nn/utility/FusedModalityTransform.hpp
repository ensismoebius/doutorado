#ifndef FUSED_MODALITY_TRANSFORM_HPP
#define FUSED_MODALITY_TRANSFORM_HPP

#include <memory>

#include "nn/utility/ITransform.hpp"

namespace nn::transforms
{

class FusedModalityTransform final : public ITransform
{
    nn::Index eeg_cols_;
    nn::Index audio_cols_;

    std::shared_ptr<ITransform> eeg_transform_;
    std::shared_ptr<ITransform> audio_transform_;

   public:
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

    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        auto out = x;

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
    } //
};

} // namespace nn::transforms

#endif // FUSED_MODALITY_TRANSFORM_HPP