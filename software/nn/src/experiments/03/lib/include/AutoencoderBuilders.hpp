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
#include <stdexcept>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "nn/initializers/kaiming_snn.hpp"
#include "nn/initializers/xavier.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/LeakyIntegrator.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/ReLU.hpp"
#include "nn/layers/ResidualBlock.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"

namespace experiment03::autoencoders
{

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
    nn::Tensor slice(input.rows(), col_count);
    for (nn::Index row = 0; row < input.rows(); ++row)
    {
        for (nn::Index col = 0; col < col_count; ++col)
        {
            slice.at(row, col) = input.at(row, col_offset + col);
        }
    }
    return slice;
}

inline auto concat_columns(const nn::Tensor& left, const nn::Tensor& right) -> nn::Tensor
{
    if (left.rows() != right.rows())
    {
        throw std::invalid_argument("concat_columns requires equal row counts");
    }

    nn::Tensor joined(left.rows(), left.cols() + right.cols());
    for (nn::Index row = 0; row < joined.rows(); ++row)
    {
        for (nn::Index col = 0; col < left.cols(); ++col)
        {
            joined.at(row, col) = left.at(row, col);
        }
        for (nn::Index col = 0; col < right.cols(); ++col)
        {
            joined.at(row, left.cols() + col) = right.at(row, col);
        }
    }
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