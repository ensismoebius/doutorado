#ifndef NN_MODELS_GRU_GRUAUTOENCODER_HPP
#define NN_MODELS_GRU_GRUAUTOENCODER_HPP

/**
 * @file include/models/gru/GRUAutoencoder.hpp
 * @brief GRU-based autoencoder for 1-D temporal signals.
 *
 * Structurally identical to nn::models::lstm::LSTMAutoencoder with the recurrent
 * layer swapped for GRULayer (no cell state). Kept as a separate model so the
 * Meeting01 comparison can report GRU-AE as a first-class baseline.
 *
 * Encoder: stacked GRULayer -> last hidden -> Linear -> tanh -> latent z
 * Decoder: latent -> Linear expand -> replicate T -> stacked GRULayer -> Linear -> recon
 *
 * The model owns hidden state; call reset_state() between independent sequences.
 */

#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "layers/Layers.hpp"
#include "layers/base/Module.hpp"
#include "layers/gru/GRULayer.hpp"
#include "models/gru/GRUAutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::gru
{

class GRUAutoencoder : public Module<nn::Backend>
{
   public:
    using Tensor = typename Module<nn::Backend>::Tensor;

    GRUAutoencoderConfig cfg_;

    std::vector<std::unique_ptr<GRULayer>> enc_grus_;
    std::unique_ptr<Linear> enc_proj_;

    std::unique_ptr<Linear> dec_expand_;
    std::vector<std::unique_ptr<GRULayer>> dec_grus_;
    std::unique_ptr<Linear> out_proj_;

    std::vector<Tensor*> param_ptrs_;

    Tensor enc_output_cache_;
    Tensor latent_cache_;
    Tensor latent_pre_cache_;
    Tensor dec_input_cache_;
    Tensor dec_output_cache_;
    Tensor recon_cache_;
    bool requires_grad_ = false;

    bool last_batched_ = false;
    int last_B_ = 1;
    int last_T_ = 0;

    explicit GRUAutoencoder(const GRUAutoencoderConfig& cfg);

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

} // namespace nn::models::gru

#endif // NN_MODELS_GRU_GRUAUTOENCODER_HPP
