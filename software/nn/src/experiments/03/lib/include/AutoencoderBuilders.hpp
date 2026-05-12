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
#include "initializers/kaiming_snn.hpp"
#include "initializers/xavier.hpp"
#include "layers/Layers.hpp"
#include "layers/convolution/Conv2d.hpp"
#include "layers/convolution/MaxPool2d.hpp"
#include "tensor/Tensor.hpp"

namespace experiment03::autoencoders
{

using nn::Leaky;
using nn::LeakyIntegrator;
using nn::LeakyReLU;
using nn::Linear;
using nn::ReLU;
using nn::ResidualBlock;
using nn::Sequential;
using nn::Tanh;

inline auto resolved_branch_hidden_size(const AutoencoderConfig& cfg) -> int;
inline auto resolved_fusion_hidden_size(const AutoencoderConfig& cfg) -> int;

struct LayerStageSpec
{
    std::string layer_type;
    std::string width_token;
    std::string activation_type;
};

enum class LayerSpecKind
{
    Linear,
    Conv1d,
    Conv2d,
    Activation,
    Residual,
    Pool1d,
    Pool2d,
    LSTM,
};

struct ParsedLayerSpec
{
    LayerSpecKind kind;
    std::string width_token;
    std::string activation_type;
    int repeat = 1;
};

inline auto split_spec_tokens(const std::string& spec) -> std::vector<std::string>
{
    std::vector<std::string> tokens;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ':'))
    {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

inline auto parse_layer_module_spec(const std::string& spec) -> ParsedLayerSpec
{
    const auto tokens = split_spec_tokens(spec);
    if (tokens.empty())
    {
        throw std::invalid_argument("Layer spec cannot be empty");
    }

    const std::string& head = tokens[0];
    if (head == "linear")
    {
        if (tokens.size() < 2 || tokens.size() > 3)
        {
            throw std::invalid_argument(
                "Linear layer spec must be 'linear:width' or 'linear:width:activation': " + spec);
        }

        ParsedLayerSpec parsed{LayerSpecKind::Linear, tokens[1], "", 1};
        if (tokens.size() == 3) parsed.activation_type = tokens[2];
        return parsed;
    }

    // Conv1D layer spec: "conv1d:out_channels:kernel_size:stride:activation"
    // Example: "conv1d:64:3:1:relu" -> Conv1D with 64 output channels, kernel 3, stride 1, ReLU
    if (head == "conv1d")
    {
        if (tokens.size() < 3 || tokens.size() > 5)
        {
            throw std::invalid_argument(
                "Conv1D layer spec must be "
                "'conv1d:out_channels:kernel_size[:stride[:activation]]': " +
                spec);
        }
        // Store in format: width_token = "out_channels:kernel:stride"
        std::stringstream conv_spec;
        conv_spec << tokens[1] << ":" << tokens[2];
        if (tokens.size() >= 4) conv_spec << ":" << tokens[3];
        ParsedLayerSpec parsed{LayerSpecKind::Conv1d, conv_spec.str(), "", 1};
        if (tokens.size() == 5) parsed.activation_type = tokens[4];
        return parsed;
    }

    // Conv2D layer spec: "conv2d:out_channels:kernel_size:stride:activation"
    // Example: "conv2d:64:3:1:relu" -> Conv2D with 64 output channels, kernel 3x3, stride 1, ReLU
    if (head == "conv2d")
    {
        if (tokens.size() < 3 || tokens.size() > 5)
        {
            throw std::invalid_argument(
                "Conv2D layer spec must be "
                "'conv2d:out_channels:kernel_size[:stride[:activation]]': " +
                spec);
        }
        std::stringstream conv_spec;
        conv_spec << tokens[1] << ":" << tokens[2];
        if (tokens.size() >= 4) conv_spec << ":" << tokens[3];
        ParsedLayerSpec parsed{LayerSpecKind::Conv2d, conv_spec.str(), "", 1};
        if (tokens.size() == 5) parsed.activation_type = tokens[4];
        return parsed;
    }

    // MaxPool1D layer spec: "pool1d:kernel_size:stride"
    // Example: "pool1d:2:2" -> MaxPool1D with kernel 2, stride 2
    if (head == "pool1d")
    {
        if (tokens.size() < 2 || tokens.size() > 3)
        {
            throw std::invalid_argument(
                "Pool1D layer spec must be 'pool1d:kernel_size[:stride]': " + spec);
        }
        ParsedLayerSpec parsed{LayerSpecKind::Pool1d, tokens[1], "", 1};
        if (tokens.size() == 3) parsed.activation_type = tokens[2]; // use as stride
        return parsed;
    }

    // MaxPool2D layer spec: "pool2d:kernel_size:stride"
    // Example: "pool2d:2:2" -> MaxPool2D with kernel 2x2, stride 2
    if (head == "pool2d")
    {
        if (tokens.size() < 2 || tokens.size() > 3)
        {
            throw std::invalid_argument(
                "Pool2D layer spec must be 'pool2d:kernel_size[:stride]': " + spec);
        }
        ParsedLayerSpec parsed{LayerSpecKind::Pool2d, tokens[1], "", 1};
        if (tokens.size() == 3) parsed.activation_type = tokens[2];
        return parsed;
    }

    if (head == "residual" || head == "residual_block")
    {
        int repeat = 1;
        if (tokens.size() == 2)
        {
            try
            {
                repeat = std::stoi(tokens[1]);
            }
            catch (const std::exception&)
            {
                throw std::invalid_argument("Residual layer repeat must be an integer: " + spec);
            }
            if (repeat < 1)
            {
                throw std::invalid_argument("Residual layer repeat must be >= 1: " + spec);
            }
        }
        else if (tokens.size() > 2)
        {
            throw std::invalid_argument(
                "Residual layer spec must be 'residual' or 'residual:N': " + spec);
        }

        return ParsedLayerSpec{LayerSpecKind::Residual, "", "", repeat};
    }

    if (head == "lstm")
    {
        if (tokens.size() < 2)
        {
            throw std::invalid_argument("LSTM layer spec must be 'lstm:hidden_size': " + spec);
        }
        return ParsedLayerSpec{LayerSpecKind::LSTM, tokens[1], "", 1};
    }

    if (tokens.size() != 1)
    {
        throw std::invalid_argument("Activation spec must be a single token: " + spec);
    }

    return ParsedLayerSpec{LayerSpecKind::Activation, "", head, 1};
}

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

inline auto build_ann_encoder(const AutoencoderConfig& cfg, int input_size, int hidden_size)
    -> Sequential
{
    if (!cfg.encoder_layer_spec.empty())
    {
        Sequential encoder;
        int current = input_size;
        for (const auto& entry : cfg.encoder_layer_spec)
        {
            const auto stage = parse_layer_module_spec(entry);
            if (stage.kind == LayerSpecKind::Linear)
            {
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
            else if (stage.kind == LayerSpecKind::Conv1d)
            {
                // Format: "out_channels:kernel:stride"
                std::stringstream ss(stage.width_token);
                int out_channels, kernel, stride = 1;
                char colon;
                ss >> out_channels >> colon >> kernel;
                if (!(ss >> colon >> stride)) stride = 1;

                auto conv = std::make_shared<Conv1dImpl<nn::Backend>>(
                    current, out_channels, kernel, stride, 1, 1);
                encoder.add_module(conv);
                append_ann_activation(encoder, stage.activation_type);
                // Update current for next layer (simplified - actual size depends on input)
                current = out_channels;
            }
            else if (stage.kind == LayerSpecKind::Conv2d)
            {
                // Format: "out_channels:kernel:stride"
                std::stringstream ss(stage.width_token);
                int out_channels, kernel, stride = 1;
                char colon;
                ss >> out_channels >> colon >> kernel;
                if (!(ss >> colon >> stride)) stride = 1;

                auto conv = std::make_shared<Conv2dImpl<nn::Backend>>(
                    current, out_channels, kernel, stride, 1, 1);
                encoder.add_module(conv);
                append_ann_activation(encoder, stage.activation_type);
                current = out_channels;
            }
            else if (stage.kind == LayerSpecKind::Pool1d)
            {
                // Format: "kernel:stride"
                std::stringstream ss(stage.width_token);
                int kernel, stride;
                char colon;
                ss >> kernel >> colon >> stride;
                if (!(ss >> colon >> stride)) stride = kernel;

                // MaxPool1d would need to be implemented - for now use fallback
                // encoder.add_module(std::make_shared<MaxPool1d>(kernel, stride));
                (void) kernel;
                (void) stride; // Placeholder
            }
            else if (stage.kind == LayerSpecKind::Pool2d)
            {
                // Format: "kernel:stride"
                std::stringstream ss(stage.width_token);
                int kernel, stride;
                char colon;
                ss >> kernel >> colon >> stride;
                if (!(ss >> colon >> stride)) stride = kernel;

                auto pool = std::make_shared<MaxPool2dImpl<nn::Backend>>(kernel, stride, 0, kernel);
                encoder.add_module(pool);
                // Output channels stay same, spatial dims change
            }
            else if (stage.kind == LayerSpecKind::Activation)
            {
                append_ann_activation(encoder, stage.activation_type);
            }
            else if (stage.kind == LayerSpecKind::LSTM)
            {
                // LSTM is handled externally by LSTMAutoencoder; skip in Sequential builder.
            }
            else
            {
                append_residual_blocks(encoder, current, stage.repeat);
            }
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
            const auto stage = parse_layer_module_spec(entry);
            if (stage.kind == LayerSpecKind::Linear)
            {
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
            else if (stage.kind == LayerSpecKind::Conv1d)
            {
                // Format: "out_channels:kernel:stride"
                std::stringstream ss(stage.width_token);
                int out_channels, kernel, stride = 1;
                char colon;
                ss >> out_channels >> colon >> kernel;
                if (!(ss >> colon >> stride)) stride = 1;

                auto conv = std::make_shared<Conv1dImpl<nn::Backend>>(
                    current, out_channels, kernel, stride, 1, 1);
                decoder.add_module(conv);
                append_ann_activation(decoder, stage.activation_type);
                current = out_channels;
            }
            else if (stage.kind == LayerSpecKind::Conv2d)
            {
                // Format: "out_channels:kernel:stride"
                std::stringstream ss(stage.width_token);
                int out_channels, kernel, stride = 1;
                char colon;
                ss >> out_channels >> colon >> kernel;
                if (!(ss >> colon >> stride)) stride = 1;

                auto conv = std::make_shared<Conv2dImpl<nn::Backend>>(
                    current, out_channels, kernel, stride, 1, 1);
                decoder.add_module(conv);
                append_ann_activation(decoder, stage.activation_type);
                current = out_channels;
            }
            else if (stage.kind == LayerSpecKind::Pool1d)
            {
                std::stringstream ss(stage.width_token);
                int kernel, stride;
                char colon;
                ss >> kernel >> colon >> stride;
                if (!(ss >> colon >> stride)) stride = kernel;
                (void) kernel;
                (void) stride; // Placeholder
            }
            else if (stage.kind == LayerSpecKind::Pool2d)
            {
                std::stringstream ss(stage.width_token);
                int kernel, stride;
                char colon;
                ss >> kernel >> colon >> stride;
                if (!(ss >> colon >> stride)) stride = kernel;

                auto pool = std::make_shared<MaxPool2dImpl<nn::Backend>>(kernel, stride, 0, kernel);
                decoder.add_module(pool);
            }
            else if (stage.kind == LayerSpecKind::Activation)
            {
                append_ann_activation(decoder, stage.activation_type);
            }
            else if (stage.kind == LayerSpecKind::LSTM)
            {
                // LSTM is handled externally by LSTMAutoencoder; skip in Sequential builder.
            }
            else
            {
                append_residual_blocks(decoder, current, stage.repeat);
            }
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
            const auto stage = parse_layer_module_spec(entry);
            if (stage.kind == LayerSpecKind::Linear)
            {
                const int output_size = resolve_width_token(cfg, stage.width_token, hidden_size);
                auto linear = std::make_shared<Linear>(current, output_size);
                kaimingSNNInitializer(linear, cfg.initializer_seed, cfg.initializer_sampler_type);
                encoder.add_module(linear);
                append_activation_by_mode(cfg, encoder, stage.activation_type, true);
                current = output_size;
            }
            else if (stage.kind == LayerSpecKind::Activation)
            {
                append_activation_by_mode(cfg, encoder, stage.activation_type, true);
            }
            else if (stage.kind == LayerSpecKind::LSTM)
            {
                // LSTM is handled externally; skip in Sequential builder.
            }
            else
            {
                append_residual_blocks(encoder, current, stage.repeat);
            }
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
            const auto stage = parse_layer_module_spec(entry);
            if (stage.kind == LayerSpecKind::Linear)
            {
                const int stage_output = resolve_width_token(cfg, stage.width_token, output_size);
                auto linear = std::make_shared<Linear>(current, stage_output);
                kaimingSNNInitializer(linear, cfg.initializer_seed, cfg.initializer_sampler_type);
                decoder.add_module(linear);
                append_activation_by_mode(cfg, decoder, stage.activation_type, true);
                current = stage_output;
            }
            else if (stage.kind == LayerSpecKind::Activation)
            {
                append_activation_by_mode(cfg, decoder, stage.activation_type, true);
            }
            else if (stage.kind == LayerSpecKind::LSTM)
            {
                // LSTM is handled externally; skip in Sequential builder.
            }
            else
            {
                append_residual_blocks(decoder, current, stage.repeat);
            }
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

using Tensor = nn::Tensor;

inline auto slice_columns(const Tensor& input, nn::Index col_offset, nn::Index col_count) -> Tensor
{
    return input.block(0, col_offset, input.rows(), col_count);
}

inline auto concat_columns(const Tensor& left, const Tensor& right) -> Tensor
{
    if (left.rows() != right.rows())
        throw std::invalid_argument("concat_columns requires equal row counts");

    Tensor joined(left.rows(), left.cols() + right.cols());
    joined.setBlock(0, 0, left);
    joined.setBlock(0, left.cols(), right);
    return joined;
}

inline auto join_params(std::initializer_list<std::vector<Tensor*>> groups) -> std::vector<Tensor*>
{
    std::vector<Tensor*> params;
    for (const auto& group : groups)
        params.insert(params.end(), group.begin(), group.end());
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