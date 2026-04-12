/**
 * @file src/experiments/03/lib/include/AutoencoderBuilders.hpp
 * @brief Autoencoderbuilders.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef EXPERIMENT03_AUTOENCODER_BUILDERS_HPP
#define EXPERIMENT03_AUTOENCODER_BUILDERS_HPP

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/initializers/kaiming_snn.hpp"
#include "nn/initializers/xavier.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

namespace experiment03::autoencoders
{

inline auto resolved_branch_hidden_size(const AutoencoderConfig& cfg) -> int;
inline auto resolved_fusion_hidden_size(const AutoencoderConfig& cfg) -> int;

struct LayerStageSpec
{
    std::string layer_type;
    std::string width_token;
    std::string activation_type;
};

inline auto parse_layer_stage_spec(const std::string& spec) -> LayerStageSpec
{
    std::stringstream ss(spec);
    std::string layer_type;
    std::string width_token;
    std::string activation_type;
    if (!std::getline(ss, layer_type, ':') || !std::getline(ss, width_token, ':') ||
        !std::getline(ss, activation_type, ':'))
    {
        throw std::invalid_argument(
            "Layer spec must use 'layer_type:width_token:activation_type': " + spec);
    }
    return {layer_type, width_token, activation_type};
}

inline auto resolve_width_token(
    const AutoencoderConfig& cfg, const std::string& width_token, int fallback_output) -> int
{
    if (width_token == "hidden") return cfg.hidden_size;
    if (width_token == "latent") return cfg.latent_size;
    if (width_token == "output") return fallback_output;
    if (width_token == "branch_hidden") return resolved_branch_hidden_size(cfg);
    if (width_token == "fusion_hidden") return resolved_fusion_hidden_size(cfg);

    try
    {
        const int width = std::stoi(width_token);
        if (width < 1)
        {
            throw std::invalid_argument("Layer width token must resolve to a positive integer");
        }
        return width;
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Unsupported width token in layer spec: " + width_token);
    }
}

inline void append_ann_activation(Sequential& seq, const std::string& activation_type)
{
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
    if (activation_type == "identity")
    {
        return;
    }

    throw std::invalid_argument("Unsupported ANN activation type: " + activation_type);
}

inline void append_snn_activation(
    const AutoencoderConfig& cfg, Sequential& seq, const std::string& activation_type)
{
    if (activation_type == "leaky")
    {
        seq.add_module(std::make_shared<Leaky>(cfg.time_step, cfg.resistance, cfg.capacitance));
        return;
    }
    if (activation_type == "leaky_integrator")
    {
        seq.add_module(
            std::make_shared<LeakyIntegrator>(cfg.time_step, cfg.resistance, cfg.capacitance));
        return;
    }
    if (activation_type == "identity")
    {
        return;
    }

    throw std::invalid_argument("Unsupported SNN activation type: " + activation_type);
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

inline auto build_ann_encoder(const AutoencoderConfig& cfg, int input_size, int hidden_size)
    -> Sequential
{
    if (!cfg.encoder_layer_spec.empty())
    {
        Sequential encoder;
        int current = input_size;
        for (const auto& entry : cfg.encoder_layer_spec)
        {
            const auto stage = parse_layer_stage_spec(entry);
            if (stage.layer_type != "linear")
            {
                throw std::invalid_argument("ANN encoder currently supports only linear stages");
            }
            const int output_size = resolve_width_token(cfg, stage.width_token, hidden_size);
            auto linear = std::make_shared<Linear>(current, output_size);
            xavierInitializer(current,
                output_size,
                linear->weight,
                linear->bias,
                cfg.initializer_seed,
                cfg.initializer_sampler_type);
            encoder.add_module(linear);
            append_ann_activation(encoder, stage.activation_type);
            current = output_size;
        }
        return encoder;
    }

    Sequential encoder;
    const auto widths = tapered_widths(cfg, hidden_size);

    int current = input_size;
    const int residual_blocks = std::max(0, cfg.residual_blocks);
    for (int width : widths)
    {
        append_ann_stage(cfg, encoder, current, width, residual_blocks);
        current = width;
    }

    auto latent_linear = std::make_shared<Linear>(current, cfg.latent_size);
    xavierInitializer(current,
        cfg.latent_size,
        latent_linear->weight,
        latent_linear->bias,
        cfg.initializer_seed,
        cfg.initializer_sampler_type);
    encoder.add_module(latent_linear);
    encoder.add_module(std::make_shared<ReLU>());
    return encoder;
}

inline auto build_ann_decoder(const AutoencoderConfig& cfg, int output_size, int hidden_size)
    -> Sequential
{
    if (!cfg.decoder_layer_spec.empty())
    {
        Sequential decoder;
        int current = cfg.latent_size;
        for (const auto& entry : cfg.decoder_layer_spec)
        {
            const auto stage = parse_layer_stage_spec(entry);
            if (stage.layer_type != "linear")
            {
                throw std::invalid_argument("ANN decoder currently supports only linear stages");
            }
            const int stage_output = resolve_width_token(cfg, stage.width_token, output_size);
            auto linear = std::make_shared<Linear>(current, stage_output);
            xavierInitializer(current,
                stage_output,
                linear->weight,
                linear->bias,
                cfg.initializer_seed,
                cfg.initializer_sampler_type);
            decoder.add_module(linear);
            append_ann_activation(decoder, stage.activation_type);
            current = stage_output;
        }
        return decoder;
    }

    Sequential decoder;
    auto widths = tapered_widths(cfg, hidden_size);
    std::reverse(widths.begin(), widths.end());

    int current = cfg.latent_size;
    const int residual_blocks = std::max(0, cfg.residual_blocks);
    for (int width : widths)
    {
        append_ann_stage(cfg, decoder, current, width, residual_blocks);
        current = width;
    }

    auto output_linear = std::make_shared<Linear>(current, output_size);
    xavierInitializer(current,
        output_size,
        output_linear->weight,
        output_linear->bias,
        cfg.initializer_seed,
        cfg.initializer_sampler_type);
    decoder.add_module(output_linear);
    return decoder;
}

inline void append_snn_stage(Sequential& seq,
    int input_size,
    int output_size,
    float time_step,
    float resistance,
    float capacitance,
    bool readout)
{
    auto linear = std::make_shared<Linear>(input_size, output_size);
    kaimingSNNInitializer(linear, std::nullopt, "");
    seq.add_module(linear);
    if (readout)
    {
        seq.add_module(std::make_shared<LeakyIntegrator>(time_step, resistance, capacitance));
    }
    else
    {
        seq.add_module(std::make_shared<Leaky>(time_step, resistance, capacitance));
    }
}

inline void append_snn_stage(const AutoencoderConfig& cfg,
    Sequential& seq,
    int input_size,
    int output_size,
    float time_step,
    float resistance,
    float capacitance,
    bool readout)
{
    auto linear = std::make_shared<Linear>(input_size, output_size);
    kaimingSNNInitializer(linear, cfg.initializer_seed, cfg.initializer_sampler_type);
    seq.add_module(linear);
    if (readout)
    {
        seq.add_module(std::make_shared<LeakyIntegrator>(time_step, resistance, capacitance));
    }
    else
    {
        seq.add_module(std::make_shared<Leaky>(time_step, resistance, capacitance));
    }
}

inline auto build_snn_encoder(const AutoencoderConfig& cfg, int input_size, int hidden_size)
    -> Sequential
{
    if (!cfg.encoder_layer_spec.empty())
    {
        Sequential encoder;
        int current = input_size;
        for (const auto& entry : cfg.encoder_layer_spec)
        {
            const auto stage = parse_layer_stage_spec(entry);
            if (stage.layer_type != "linear")
            {
                throw std::invalid_argument("SNN encoder currently supports only linear stages");
            }
            const int output_size = resolve_width_token(cfg, stage.width_token, hidden_size);
            auto linear = std::make_shared<Linear>(current, output_size);
            kaimingSNNInitializer(linear, cfg.initializer_seed, cfg.initializer_sampler_type);
            encoder.add_module(linear);
            append_snn_activation(cfg, encoder, stage.activation_type);
            current = output_size;
        }
        return encoder;
    }

    Sequential encoder;
    const auto widths = tapered_widths(cfg, hidden_size);

    int current = input_size;
    for (int width : widths)
    {
        append_snn_stage(
            cfg, encoder, current, width, cfg.time_step, cfg.resistance, cfg.capacitance, false);
        current = width;
    }

    append_snn_stage(cfg,
        encoder,
        current,
        cfg.latent_size,
        cfg.time_step,
        cfg.resistance,
        cfg.capacitance,
        false);
    return encoder;
}

inline auto build_snn_decoder(const AutoencoderConfig& cfg, int output_size, int hidden_size)
    -> Sequential
{
    if (!cfg.decoder_layer_spec.empty())
    {
        Sequential decoder;
        int current = cfg.latent_size;
        for (const auto& entry : cfg.decoder_layer_spec)
        {
            const auto stage = parse_layer_stage_spec(entry);
            if (stage.layer_type != "linear")
            {
                throw std::invalid_argument("SNN decoder currently supports only linear stages");
            }
            const int stage_output = resolve_width_token(cfg, stage.width_token, output_size);
            auto linear = std::make_shared<Linear>(current, stage_output);
            kaimingSNNInitializer(linear, cfg.initializer_seed, cfg.initializer_sampler_type);
            decoder.add_module(linear);
            append_snn_activation(cfg, decoder, stage.activation_type);
            current = stage_output;
        }
        return decoder;
    }

    Sequential decoder;
    auto widths = tapered_widths(cfg, hidden_size);
    std::reverse(widths.begin(), widths.end());

    int current = cfg.latent_size;
    for (int width : widths)
    {
        append_snn_stage(
            cfg, decoder, current, width, cfg.time_step, cfg.resistance, cfg.capacitance, true);
        current = width;
    }

    auto output_linear = std::make_shared<Linear>(current, output_size);
    kaimingSNNInitializer(output_linear, cfg.initializer_seed, cfg.initializer_sampler_type);
    decoder.add_module(output_linear);
    decoder.add_module(
        std::make_shared<LeakyIntegrator>(cfg.time_step, cfg.resistance, cfg.capacitance));
    return decoder;
}

inline auto slice_columns(const nn::Tensor& input, nn::Index col_offset, nn::Index col_count)
    -> nn::Tensor
{
    return input.block(0, col_offset, input.rows(), col_count);
}

inline auto concat_columns(const nn::Tensor& left, const nn::Tensor& right) -> nn::Tensor
{
    if (left.rows() != right.rows())
    {
        throw std::invalid_argument("concat_columns requires equal row counts");
    }

    nn::Tensor joined(left.rows(), left.cols() + right.cols());
    joined.setBlock(0, 0, left);
    joined.setBlock(0, left.cols(), right);
    return joined;
}

inline auto join_params(std::initializer_list<std::vector<nn::Tensor*>> groups)
    -> std::vector<nn::Tensor*>
{
    std::vector<nn::Tensor*> params;
    for (const auto& group : groups)
    {
        params.insert(params.end(), group.begin(), group.end());
    }
    return params;
}

inline void reset_sequential_state(Sequential& seq)
{
    for (auto& layer : seq.layers)
    {
        layer->reset_state();
    }
}

} // namespace experiment03::autoencoders

#endif // EXPERIMENT03_AUTOENCODER_BUILDERS_HPP