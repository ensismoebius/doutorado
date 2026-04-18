/**
 * @file LSTMAutoencoder.cpp
 * @brief LSTM Autoencoder — forward and BPTT backward implementation.
 */

#include "LSTMAutoencoder.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "nn/initializers/xavier.hpp"

namespace experiment04
{

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
LSTMAutoencoder::LSTMAutoencoder(const LSTMAutoencoderConfig& cfg) : cfg_(cfg)
{
    const int D = cfg_.input_size;
    const int H = cfg_.hidden_size;
    const int Z = cfg_.latent_size;
    const int L = cfg_.num_layers;

    // ---- Encoder LSTM stack ----
    for (int l = 0; l < L; ++l)
    {
        int in_dim = (l == 0) ? D : H;
        enc_lstms_.push_back(std::make_unique<LSTMLayer>(in_dim, H));
    }

    // Encoder projection  H → Z  (with tanh applied manually in encode())
    enc_proj_ = std::make_unique<Linear>(H, Z);
    xavierInitializer(H, Z, enc_proj_->weight, enc_proj_->bias, 100u);

    // ---- Decoder expansion  Z → H ----
    dec_expand_ = std::make_unique<Linear>(Z, H);
    xavierInitializer(Z, H, dec_expand_->weight, dec_expand_->bias, 101u);

    // ---- Decoder LSTM stack ----
    // The decoder LSTM receives repeated h_expand : [T × H] as its input
    for (int l = 0; l < L; ++l)
    {
        dec_lstms_.push_back(std::make_unique<LSTMLayer>(H, H));
    }

    // Output projection  H → D
    out_proj_ = std::make_unique<Linear>(H, D);
    xavierInitializer(H, D, out_proj_->weight, out_proj_->bias, 102u);

    build_param_ptrs();
}

// ---------------------------------------------------------------------------
// build_param_ptrs
// ---------------------------------------------------------------------------
void LSTMAutoencoder::build_param_ptrs()
{
    param_ptrs_.clear();
    for (auto& lstm : enc_lstms_)
    {
        for (nn::Tensor* p : lstm->params()) param_ptrs_.push_back(p);
    }
    for (nn::Tensor* p : enc_proj_->params()) param_ptrs_.push_back(p);
    for (nn::Tensor* p : dec_expand_->params()) param_ptrs_.push_back(p);
    for (auto& lstm : dec_lstms_)
    {
        for (nn::Tensor* p : lstm->params()) param_ptrs_.push_back(p);
    }
    for (nn::Tensor* p : out_proj_->params()) param_ptrs_.push_back(p);
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
auto LSTMAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    // input : [T × D]
    Tensor h = input;
    for (auto& lstm : enc_lstms_)
    {
        lstm->reset_state();
        h = lstm->forward(h, requires_grad); // [T × H]
    }
    enc_output_cache_ = h;

    // Take only the last time step's hidden state: [1 × H]
    const int T = static_cast<int>(h.rows());
    Tensor h_last = h.row(static_cast<nn::Index>(T - 1)); // [1 × H]

    // Project to latent space
    Tensor z_pre = enc_proj_->forward(h_last, requires_grad); // [1 × Z]
    latent_pre_cache_ = z_pre;

    // Bounded latent codes via tanh
    Tensor z = tanh_tensor(z_pre); // [1 × Z]
    latent_cache_ = z;

    return z; // [1 × Z]
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
auto LSTMAutoencoder::decode(const Tensor& latent, int seq_len, bool requires_grad) -> Tensor
{
    // latent : [1 × Z]
    // Expand latent to hidden dim
    Tensor h_expand = dec_expand_->forward(latent, requires_grad); // [1 × H]

    // Repeat h_expand for seq_len steps to form the decoder input [T × H]
    const int H = cfg_.hidden_size;
    Tensor dec_in(static_cast<nn::Index>(seq_len), static_cast<nn::Index>(H));
    for (int t = 0; t < seq_len; ++t)
    {
        for (nn::Index col = 0; col < static_cast<nn::Index>(H); ++col)
        {
            dec_in.at(static_cast<nn::Index>(t), col) = h_expand.at(0, col);
        }
    }
    dec_input_cache_ = dec_in;

    // Pass through decoder LSTM stack
    Tensor dec_h = dec_in;
    for (auto& lstm : dec_lstms_)
    {
        lstm->reset_state();
        dec_h = lstm->forward(dec_h, requires_grad); // [T × H]
    }
    dec_output_cache_ = dec_h;

    // Project each time step to input dimension
    // out_proj_ expects [batch × H]; we pass [T × H] treating T as batch
    Tensor recon = out_proj_->forward(dec_h, requires_grad); // [T × D]
    recon_cache_ = recon;

    return recon;
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------
auto LSTMAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    requires_grad_ = requires_grad;
    const int T = static_cast<int>(input.rows());

    Tensor z = encode(input, requires_grad);
    return decode(z, T, requires_grad);
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------
auto LSTMAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    // grad_output : [T × D]  dL/d_recon

    // ---- 1. Output projection backward  [T × D] → [T × H] ----
    Tensor d_dec_h = out_proj_->backward(grad_output); // [T × H]

    // ---- 2. Decoder LSTM stack backward (reversed) ----
    Tensor d_dec_in = d_dec_h;
    for (int l = static_cast<int>(dec_lstms_.size()) - 1; l >= 0; --l)
    {
        d_dec_in = dec_lstms_[static_cast<size_t>(l)]->backward(d_dec_in); // [T × H]
    }

    // ---- 3. Sum gradient over repeated time steps → [1 × H] ----
    //         d_dec_in : [T × H] — each row corresponds to one copy of h_expand,
    //         so total gradient is the sum over all T rows.
    Tensor d_h_expand = nn::Tensor::zeros(1, cfg_.hidden_size);
    for (int t = 0; t < static_cast<int>(d_dec_in.rows()); ++t)
    {
        for (nn::Index col = 0; col < static_cast<nn::Index>(cfg_.hidden_size); ++col)
        {
            d_h_expand.at(0, col) += d_dec_in.at(static_cast<nn::Index>(t), col);
        }
    }

    // ---- 4. Decoder expand backward  [1 × H] → [1 × Z] ----
    Tensor d_z = dec_expand_->backward(d_h_expand); // [1 × Z]

    // ---- 5. tanh backward through latent projection ----
    //         d/dz_pre = d_z ⊙ (1 - tanh²(z_pre)) = d_z ⊙ tanh_grad(z)
    Tensor d_z_pre = d_z * tanh_grad(latent_cache_); // [1 × Z]

    // ---- 6. Encoder projection backward  [1 × Z] → [1 × H] ----
    Tensor d_h_last = enc_proj_->backward(d_z_pre); // [1 × H]

    // ---- 7. Propagate gradient to encoder LSTMs ----
    //         Only the last hidden state receives gradient; create [T × H] with
    //         gradient only in the last row.
    const int T = static_cast<int>(enc_output_cache_.rows());
    Tensor d_enc_h =
        nn::Tensor::zeros(static_cast<nn::Index>(T), static_cast<nn::Index>(cfg_.hidden_size));
    for (nn::Index col = 0; col < static_cast<nn::Index>(cfg_.hidden_size); ++col)
    {
        d_enc_h.at(static_cast<nn::Index>(T - 1), col) = d_h_last.at(0, col);
    }

    // ---- 8. Encoder LSTM stack backward (reversed) ----
    Tensor d_input = d_enc_h;
    for (int l = static_cast<int>(enc_lstms_.size()) - 1; l >= 0; --l)
    {
        d_input = enc_lstms_[static_cast<size_t>(l)]->backward(d_input); // [T × D or T × H]
    }

    return d_input; // [T × D]
}

// ---------------------------------------------------------------------------
// reset_state
// ---------------------------------------------------------------------------
void LSTMAutoencoder::reset_state()
{
    for (auto& lstm : enc_lstms_) lstm->reset_state();
    for (auto& lstm : dec_lstms_) lstm->reset_state();
}

// ---------------------------------------------------------------------------
// params
// ---------------------------------------------------------------------------
auto LSTMAutoencoder::params() -> std::span<nn::Tensor*>
{
    return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}

// ---------------------------------------------------------------------------
// state_dict / load_state_dict
// ---------------------------------------------------------------------------
auto LSTMAutoencoder::state_dict() const -> std::map<std::string, nn::Tensor>
{
    std::map<std::string, nn::Tensor> sd;

    auto prefix_merge = [&](const std::map<std::string, nn::Tensor>& src, const std::string& pfx)
    {
        for (const auto& [k, v] : src) sd[pfx + k] = v;
    };

    for (size_t l = 0; l < enc_lstms_.size(); ++l)
    {
        prefix_merge(enc_lstms_[l]->state_dict(), "enc_lstm" + std::to_string(l) + ".");
    }
    prefix_merge(enc_proj_->state_dict(), "enc_proj.");
    prefix_merge(dec_expand_->state_dict(), "dec_expand.");
    for (size_t l = 0; l < dec_lstms_.size(); ++l)
    {
        prefix_merge(dec_lstms_[l]->state_dict(), "dec_lstm" + std::to_string(l) + ".");
    }
    prefix_merge(out_proj_->state_dict(), "out_proj.");
    return sd;
}

void LSTMAutoencoder::load_state_dict(const std::map<std::string, nn::Tensor>& sd)
{
    auto extract_prefix = [&](const std::string& pfx) -> std::map<std::string, nn::Tensor>
    {
        std::map<std::string, nn::Tensor> sub;
        for (const auto& [k, v] : sd)
        {
            if (k.substr(0, pfx.size()) == pfx)
            {
                sub[k.substr(pfx.size())] = v;
            }
        }
        return sub;
    };

    for (size_t l = 0; l < enc_lstms_.size(); ++l)
    {
        enc_lstms_[l]->load_state_dict(extract_prefix("enc_lstm" + std::to_string(l) + "."));
    }
    enc_proj_->load_state_dict(extract_prefix("enc_proj."));
    dec_expand_->load_state_dict(extract_prefix("dec_expand."));
    for (size_t l = 0; l < dec_lstms_.size(); ++l)
    {
        dec_lstms_[l]->load_state_dict(extract_prefix("dec_lstm" + std::to_string(l) + "."));
    }
    out_proj_->load_state_dict(extract_prefix("out_proj."));
}

} // namespace experiment04
