#ifndef NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODER_HPP
#define NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODER_HPP

/**
 * @file include/models/transformer/TransformerAutoencoder.hpp
 * @brief Bottlenecked Transformer autoencoder for 1-D temporal signals.
 *
 * Design choice: there is NO encoder-decoder cross-attention. The decoder is
 * conditioned only on the fixed-dimension latent z, so all reconstruction
 * information must pass through the bottleneck (a strict-compression AE), which
 * makes it directly comparable to PCA / the recurrent AEs.
 *
 * Encoder: Linear(D->d_model) + positional -> N encoder blocks
 *          -> mean-pool over time -> Linear(d_model->Z) -> tanh -> z
 * Decoder: Linear(Z->d_model) -> replicate T -> + positional
 *          -> N encoder blocks (self-attention only) -> Linear(d_model->D) per step
 *
 * 2-D single-sequence contract only (Guayaquil trains at batch size 1):
 *   forward(Tensor{T, D}) -> Tensor{T, D}
 *
 * Every constituent layer is finite-difference gradient-checked; the composed
 * model has its own check in transformer_autoencoder_gtest.cpp.
 */

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "layers/Layers.hpp"
#include "layers/attention/TransformerEncoderBlock.hpp"
#include "layers/base/Module.hpp"
#include "models/transformer/TransformerAutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::transformer
{

class TransformerAutoencoder : public Module<nn::Backend>
{
   public:
    using Tensor = typename Module<nn::Backend>::Tensor;
    using Block = TransformerEncoderBlockImpl<nn::Backend>;

    TransformerAutoencoderConfig cfg_;

    std::unique_ptr<Linear> embed_;
    std::vector<std::unique_ptr<Block>> enc_blocks_;
    std::unique_ptr<Linear> to_latent_;

    std::unique_ptr<Linear> from_latent_;
    std::vector<std::unique_ptr<Block>> dec_blocks_;
    std::unique_ptr<Linear> out_proj_;

    Tensor pe_; // (T, d_model) fixed sinusoidal positional encoding
    Tensor latent_cache_;
    int last_T_ = 0;
    bool forward_cached_ = false;

    std::vector<Tensor*> param_ptrs_;

    explicit TransformerAutoencoder(const TransformerAutoencoderConfig& cfg);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, int seq_len, bool requires_grad = true) -> Tensor;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;

    void reset_state() override;
    auto params() -> std::span<Tensor*> override;
    auto state_dict() const -> std::map<std::string, Tensor> override;
    void load_state_dict(const std::map<std::string, Tensor>& sd) override;

   private:
    void build_param_ptrs();
};

} // namespace nn::models::transformer

#endif // NN_MODELS_TRANSFORMER_TRANSFORMERAUTOENCODER_HPP
