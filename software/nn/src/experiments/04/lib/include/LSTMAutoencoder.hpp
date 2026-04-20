#pragma once

/**
 * @file LSTMAutoencoder.hpp
 * @brief LSTM-based autoencoder for 1-D temporal signals.
 */

#include <memory>
#include <span>
#include <vector>

#include "LSTMLayer.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

namespace lstm_autoencoder_experiment
{

struct LSTMAutoencoderConfig
{
    int input_size = 64;
    int seq_len = 32;
    int hidden_size = 128;
    int latent_size = 16;
    int num_layers = 1;
};

class LSTMAutoencoder : public Module<nn::EigenTensorBackend>
{
   public:
    using Tensor = nn::Tensor;

    LSTMAutoencoderConfig cfg_;
    std::vector<std::unique_ptr<LSTMLayer>> enc_lstms_;
    std::unique_ptr<Linear> enc_proj_;
    std::unique_ptr<Linear> dec_expand_;
    std::vector<std::unique_ptr<LSTMLayer>> dec_lstms_;
    std::unique_ptr<Linear> out_proj_;
    std::vector<nn::Tensor*> param_ptrs_;
    Tensor enc_output_cache_;
    Tensor latent_cache_;
    Tensor latent_pre_cache_;
    Tensor dec_input_cache_;
    Tensor dec_output_cache_;
    Tensor recon_cache_;
    bool requires_grad_ = false;

    explicit LSTMAutoencoder(const LSTMAutoencoderConfig& cfg);

    auto encode(const Tensor& input, bool requires_grad = true) -> Tensor;
    auto decode(const Tensor& latent, int seq_len, bool requires_grad = true) -> Tensor;
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override;
    auto backward(const Tensor& grad_output) -> Tensor override;
    void reset_state() override;
    auto params() -> std::span<nn::Tensor*> override;
    auto state_dict() const -> std::map<std::string, nn::Tensor> override;
    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override;

   private:
    void build_param_ptrs();
};

} // namespace lstm_autoencoder_experiment