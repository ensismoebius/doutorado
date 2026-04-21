/**
 * @file LSTMAutoencoder.cpp
 * @brief LSTM autoencoder forward and BPTT backward implementation.
 */

#include "LSTMAutoencoder.hpp"

#include <random>
#include <string>

namespace lstm_autoencoder_experiment
{

LSTMAutoencoder::LSTMAutoencoder(const LSTMAutoencoderConfig& cfg) : cfg_(cfg)
{
    const int D = cfg_.input_size;
    const int H = cfg_.hidden_size;
    const int Z = cfg_.latent_size;
    const int L = cfg_.num_layers;

    for (int l = 0; l < L; ++l)
    {
        const int in_dim = (l == 0) ? D : H;
        enc_lstms_.push_back(std::make_unique<LSTMLayer>(in_dim, H));
    }

    auto normal_fill = [](nn::Tensor& t, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 0.05f);
        for (nn::Index k = 0; k < static_cast<nn::Index>(t.size()); ++k)
        {
            t.at(k) = dist(rng);
        }
    };

    enc_proj_ = std::make_unique<Linear>(H, Z);
    normal_fill(enc_proj_->weight, 100u);
    enc_proj_->bias.set_zero();

    dec_expand_ = std::make_unique<Linear>(Z, H);
    normal_fill(dec_expand_->weight, 101u);
    dec_expand_->bias.set_zero();

    for (int l = 0; l < L; ++l)
    {
        dec_lstms_.push_back(std::make_unique<LSTMLayer>(H, H));
    }

    out_proj_ = std::make_unique<Linear>(H, D);
    normal_fill(out_proj_->weight, 102u);
    out_proj_->bias.set_zero();

    build_param_ptrs();
}

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

auto LSTMAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    Tensor h = input;
    for (auto& lstm : enc_lstms_)
    {
        lstm->reset_state();
        h = lstm->forward(h, requires_grad);
    }
    enc_output_cache_ = h;

    const int T = static_cast<int>(h.rows());
    Tensor h_last = h.row(static_cast<nn::Index>(T - 1));
    Tensor z_pre = enc_proj_->forward(h_last, requires_grad);
    latent_pre_cache_ = z_pre;
    Tensor z = tanh_tensor(z_pre);
    latent_cache_ = z;
    return z;
}

auto LSTMAutoencoder::decode(const Tensor& latent, int seq_len, bool requires_grad) -> Tensor
{
    Tensor h_expand = dec_expand_->forward(latent, requires_grad);

    Tensor dec_in = nn::Tensor::ones(static_cast<nn::Index>(seq_len), 1).matmul(h_expand);
    dec_input_cache_ = dec_in;

    Tensor dec_h = dec_in;
    for (auto& lstm : dec_lstms_)
    {
        lstm->reset_state();
        dec_h = lstm->forward(dec_h, requires_grad);
    }
    dec_output_cache_ = dec_h;

    Tensor recon = out_proj_->forward(dec_h, requires_grad);
    recon_cache_ = recon;
    return recon;
}

auto LSTMAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    requires_grad_ = requires_grad;
    const int T = static_cast<int>(input.rows());
    Tensor z = encode(input, requires_grad);
    return decode(z, T, requires_grad);
}

auto LSTMAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    Tensor d_dec_h = out_proj_->backward(grad_output);

    Tensor d_dec_in = d_dec_h;
    for (int l = static_cast<int>(dec_lstms_.size()) - 1; l >= 0; --l)
    {
        d_dec_in = dec_lstms_[static_cast<size_t>(l)]->backward(d_dec_in);
    }

    Tensor d_h_expand = nn::Tensor::ones(1, d_dec_in.rows()).matmul(d_dec_in);

    Tensor d_z = dec_expand_->backward(d_h_expand);
    Tensor d_z_pre = d_z * tanh_grad(latent_cache_);
    Tensor d_h_last = enc_proj_->backward(d_z_pre);

    const int T = static_cast<int>(enc_output_cache_.rows());
    Tensor d_enc_h =
        nn::Tensor::zeros(static_cast<nn::Index>(T), static_cast<nn::Index>(cfg_.hidden_size));
    d_enc_h.setBlock(static_cast<nn::Index>(T - 1), 0, d_h_last);

    Tensor d_input = d_enc_h;
    for (int l = static_cast<int>(enc_lstms_.size()) - 1; l >= 0; --l)
    {
        d_input = enc_lstms_[static_cast<size_t>(l)]->backward(d_input);
    }

    return d_input;
}

void LSTMAutoencoder::reset_state()
{
    for (auto& lstm : enc_lstms_) lstm->reset_state();
    for (auto& lstm : dec_lstms_) lstm->reset_state();
}

auto LSTMAutoencoder::params() -> std::span<nn::Tensor*>
{
    return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}

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

} // namespace lstm_autoencoder_experiment