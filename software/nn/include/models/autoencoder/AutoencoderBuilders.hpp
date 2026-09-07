/**
 * @file include/models/autoencoder/AutoencoderBuilders.hpp
 * @brief Autoencoderbuilders.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 *
 * This is the thin public entry point: `LayerStageSpec`/`ParsedLayerSpec` plus
 * the 4 top-level build_* functions. The spec-string parsing helpers live in
 * AutoencoderLayerSpecParsers.hpp, the per-stage layer-appending helpers in
 * AutoencoderStageBuilders.hpp, and small tensor/param plumbing utilities in
 * AutoencoderTensorUtils.hpp — all included below.
 */

#ifndef NN_MODELS_AUTOENCODER_AUTOENCODER_BUILDERS_HPP
#define NN_MODELS_AUTOENCODER_AUTOENCODER_BUILDERS_HPP

#include <sstream>
#include <string>
#include <vector>

#include "layers/Layers.hpp"
#include "layers/convolution/Conv2d.hpp"
#include "layers/convolution/MaxPool2d.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "tensor/Tensor.hpp"

namespace nn::models::autoencoder
{

using nn::Linear;
using nn::Sequential;

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

} // namespace nn::models::autoencoder

#include "models/autoencoder/AutoencoderLayerSpecParsers.hpp"
#include "models/autoencoder/AutoencoderStageBuilders.hpp"
#include "models/autoencoder/AutoencoderTensorUtils.hpp"

namespace nn::models::autoencoder
{

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
            cfg, encoder, current, width, cfg.delta_t, cfg.resistance, cfg.capacitance, false);
        current = width;
    }

    append_snn_stage(cfg,
        encoder,
        current,
        cfg.latent_size,
        cfg.delta_t,
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
            cfg, decoder, current, width, cfg.delta_t, cfg.resistance, cfg.capacitance, true);
        current = width;
    }

    auto output_linear = std::make_shared<Linear>(current, output_size);
    kaimingSNNInitializer(output_linear, cfg.initializer_seed, cfg.initializer_sampler_type);
    decoder.add_module(output_linear);
    require_time_steps(cfg.time_steps);
    decoder.add_module(std::make_shared<LifBPTT>(cfg.time_steps,
        cfg.delta_t,
        cfg.resistance,
        cfg.capacitance,
        /*voltage_threshold=*/1.0F,
        /*reset_zero=*/true,
        /*reset_potential=*/0.0F,
        /*readout_mode=*/true));
    return decoder;
}

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUTOENCODER_BUILDERS_HPP
