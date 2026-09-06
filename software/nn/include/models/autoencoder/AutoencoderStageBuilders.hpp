/**
 * @file include/models/autoencoder/AutoencoderStageBuilders.hpp
 * @brief "Append one stage of layers to a builder" helpers used by the
 *        build_ann_encoder/build_ann_decoder/build_snn_encoder/build_snn_decoder
 *        entry points (extracted from AutoencoderBuilders.hpp).
 */

#ifndef NN_MODELS_AUTOENCODER_AUTOENCODER_STAGE_BUILDERS_HPP
#define NN_MODELS_AUTOENCODER_AUTOENCODER_STAGE_BUILDERS_HPP

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "initializers/kaiming_snn.hpp"
#include "initializers/xavier.hpp"
#include "layers/Layers.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"

namespace nn::models::autoencoder
{

using nn::LeakyReLU;
using nn::Lif;
using nn::LifBPTT;
using nn::LifIntegrator;
using nn::Linear;
using nn::ReLU;
using nn::ResidualBlock;
using nn::Sequential;
using nn::Tanh;

inline void append_ann_activation(Sequential& seq, const std::string& activation_type)
{
    if (activation_type.empty())
    {
        return;
    }

    if (activation_type == "relu")
    {
        seq.add_module(std::make_shared<ReLU>());
        return;
    }
    if (activation_type == "leaky_relu")
    {
        seq.add_module(std::make_shared<LeakyReLU>());
        return;
    }
    if (activation_type == "leaky")
    {
        seq.add_module(std::make_shared<LeakyReLU>());
        return;
    }
    if (activation_type == "tanh")
    {
        seq.add_module(std::make_shared<nn::Tanh>());
        return;
    }
    if (activation_type == "identity")
    {
        return;
    }

    throw std::invalid_argument("Unsupported ANN activation type: " + activation_type);
}

// Reject an unset/invalid BPTT sequence length instead of assuming 1. Assuming 1 would
// silently turn LifBPTT into a single-step Lif — a model that trains, reports a loss,
// and has no temporal credit assignment whatsoever.
inline void require_time_steps(int time_steps)
{
    if (time_steps < 1)
        throw std::invalid_argument(
            "AutoencoderBuilders: AutoencoderConfig::time_steps is unset (" +
            std::to_string(time_steps) +
            "). A spiking autoencoder must declare its BPTT sequence length explicitly; "
            "defaulting it to 1 would silently build a single-step network with no "
            "temporal credit assignment. Set cfg.time_steps to the number of frames per "
            "sample.");
}

inline void append_snn_activation(
    const AutoencoderConfig& cfg, Sequential& seq, const std::string& activation_type)
{
    if (activation_type == "leaky")
    {
        require_time_steps(cfg.time_steps);
        seq.add_module(std::make_shared<LifBPTT>(
            cfg.time_steps, cfg.delta_t, cfg.resistance, cfg.capacitance, cfg.voltage_threshold));
        return;
    }
    if (activation_type == "leaky_integrator")
    {
        // Readout (decoder) neuron: emits v_mem directly, no spike/reset. Same BPTT
        // unroll as the encoder so the whole stack shares one time axis.
        require_time_steps(cfg.time_steps);
        seq.add_module(std::make_shared<LifBPTT>(cfg.time_steps,
            cfg.delta_t,
            cfg.resistance,
            cfg.capacitance,
            /*voltage_threshold=*/1.0F,
            /*reset_zero=*/true,
            /*reset_potential=*/0.0F,
            /*readout_mode=*/true));
        return;
    }
    if (activation_type == "identity")
    {
        return;
    }

    throw std::invalid_argument("Unsupported SNN activation type: " + activation_type);
}

inline void append_residual_blocks(Sequential& seq, int features, int repeat)
{
    for (int i = 0; i < repeat; ++i)
    {
        seq.add_module(std::make_shared<ResidualBlock>(features));
    }
}

inline void append_activation_by_mode(const AutoencoderConfig& cfg,
    Sequential& seq,
    const std::string& activation_type,
    bool snn_mode)
{
    if (activation_type.empty()) return;

    if (!snn_mode)
    {
        append_ann_activation(seq, activation_type);
        return;
    }

    if (activation_type == "leaky" || activation_type == "leaky_integrator" ||
        activation_type == "identity")
    {
        append_snn_activation(cfg, seq, activation_type);
        return;
    }

    append_ann_activation(seq, activation_type);
}

inline auto tapered_widths(const AutoencoderConfig& cfg, int base_hidden) -> std::vector<int>
{
    if (!cfg.layer_sizes.empty())
    {
        std::vector<int> widths;
        widths.reserve(cfg.layer_sizes.size());
        for (int w : cfg.layer_sizes)
        {
            if (w < 1)
            {
                throw std::invalid_argument(
                    "AutoencoderConfig::layer_sizes must contain positive values");
            }
            widths.push_back(w);
        }
        return widths;
    }

    if (cfg.depth < 1)
    {
        throw std::invalid_argument("AutoencoderConfig::depth must be >= 1");
    }

    std::vector<int> widths;
    widths.reserve(static_cast<std::size_t>(cfg.depth));

    int current = std::max(base_hidden, cfg.latent_size * 2);
    widths.push_back(current);
    for (int i = 1; i < cfg.depth; ++i)
    {
        current = std::max(cfg.latent_size * 2, current / 2);
        widths.push_back(current);
    }

    return widths;
}

inline auto resolved_branch_hidden_size(const AutoencoderConfig& cfg) -> int
{
    if (cfg.branch_hidden_size > 0)
    {
        return cfg.branch_hidden_size;
    }
    return std::max(cfg.hidden_size, cfg.latent_size * 2);
}

inline auto resolved_fusion_hidden_size(const AutoencoderConfig& cfg) -> int
{
    if (cfg.fusion_hidden_size > 0)
    {
        return cfg.fusion_hidden_size;
    }
    return std::max(cfg.hidden_size, cfg.latent_size * 3);
}

inline void append_ann_stage(Sequential& seq, int input_size, int output_size, int residual_blocks)
{
    auto linear = std::make_shared<Linear>(input_size, output_size);
    xavierInitializer(input_size, output_size, linear->weight, linear->bias, std::nullopt, "");
    seq.add_module(linear);
    seq.add_module(std::make_shared<ReLU>());
    for (int i = 0; i < residual_blocks; ++i)
    {
        seq.add_module(std::make_shared<ResidualBlock>(output_size));
    }
}

inline void append_ann_stage(const AutoencoderConfig& cfg,
    Sequential& seq,
    int input_size,
    int output_size,
    int residual_blocks)
{
    auto linear = std::make_shared<Linear>(input_size, output_size);
    xavierInitializer(input_size,
        output_size,
        linear->weight,
        linear->bias,
        cfg.initializer_seed,
        cfg.initializer_sampler_type);
    seq.add_module(linear);
    seq.add_module(std::make_shared<ReLU>());
    for (int i = 0; i < residual_blocks; ++i)
    {
        seq.add_module(std::make_shared<ResidualBlock>(output_size));
    }
}

inline void append_snn_stage(Sequential& seq,
    int input_size,
    int output_size,
    int time_steps,
    float delta_t,
    float resistance,
    float capacitance,
    bool readout)
{
    require_time_steps(time_steps);
    auto linear = std::make_shared<Linear>(input_size, output_size);
    kaimingSNNInitializer(linear, std::nullopt, "");
    seq.add_module(linear);
    if (readout)
    {
        seq.add_module(std::make_shared<LifBPTT>(time_steps,
            delta_t,
            resistance,
            capacitance,
            /*voltage_threshold=*/1.0F,
            /*reset_zero=*/true,
            /*reset_potential=*/0.0F,
            /*readout_mode=*/true));
    }
    else
    {
        seq.add_module(std::make_shared<LifBPTT>(time_steps, delta_t, resistance, capacitance));
    }
}

inline void append_snn_stage(const AutoencoderConfig& cfg,
    Sequential& seq,
    int input_size,
    int output_size,
    float delta_t,
    float resistance,
    float capacitance,
    bool readout)
{
    require_time_steps(cfg.time_steps);
    auto linear = std::make_shared<Linear>(input_size, output_size);
    kaimingSNNInitializer(linear, cfg.initializer_seed, cfg.initializer_sampler_type);
    seq.add_module(linear);
    if (readout)
    {
        seq.add_module(std::make_shared<LifBPTT>(cfg.time_steps,
            delta_t,
            resistance,
            capacitance,
            /*voltage_threshold=*/1.0F,
            /*reset_zero=*/true,
            /*reset_potential=*/0.0F,
            /*readout_mode=*/true));
    }
    else
    {
        seq.add_module(std::make_shared<LifBPTT>(cfg.time_steps, delta_t, resistance, capacitance));
    }
}

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUTOENCODER_STAGE_BUILDERS_HPP
