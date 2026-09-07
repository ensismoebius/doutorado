/**
 * @file include/models/autoencoder/AutoencoderLayerSpecParsers.hpp
 * @brief Spec-string-to-struct parsing layer for autoencoder layer specs
 *        (extracted from AutoencoderBuilders.hpp).
 */

#ifndef NN_MODELS_AUTOENCODER_AUTOENCODER_LAYER_SPEC_PARSERS_HPP
#define NN_MODELS_AUTOENCODER_AUTOENCODER_LAYER_SPEC_PARSERS_HPP

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/autoencoder/AutoencoderBuilders.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "models/autoencoder/AutoencoderStageBuilders.hpp"

namespace nn::models::autoencoder
{

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

inline auto parse_linear_layer_spec(const std::vector<std::string>& tokens, const std::string& spec)
    -> ParsedLayerSpec
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
inline auto parse_conv1d_layer_spec(const std::vector<std::string>& tokens, const std::string& spec)
    -> ParsedLayerSpec
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
inline auto parse_conv2d_layer_spec(const std::vector<std::string>& tokens, const std::string& spec)
    -> ParsedLayerSpec
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
inline auto parse_pool1d_layer_spec(const std::vector<std::string>& tokens, const std::string& spec)
    -> ParsedLayerSpec
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
inline auto parse_pool2d_layer_spec(const std::vector<std::string>& tokens, const std::string& spec)
    -> ParsedLayerSpec
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

inline auto parse_residual_layer_spec(
    const std::vector<std::string>& tokens, const std::string& spec) -> ParsedLayerSpec
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

inline auto parse_lstm_layer_spec(const std::vector<std::string>& tokens, const std::string& spec)
    -> ParsedLayerSpec
{
    if (tokens.size() < 2)
    {
        throw std::invalid_argument("LSTM layer spec must be 'lstm:hidden_size': " + spec);
    }
    return ParsedLayerSpec{LayerSpecKind::LSTM, tokens[1], "", 1};
}

inline auto parse_layer_module_spec(const std::string& spec) -> ParsedLayerSpec
{
    const auto tokens = split_spec_tokens(spec);
    if (tokens.empty())
    {
        throw std::invalid_argument("Layer spec cannot be empty");
    }

    const std::string& head = tokens[0];
    if (head == "linear") return parse_linear_layer_spec(tokens, spec);
    if (head == "conv1d") return parse_conv1d_layer_spec(tokens, spec);
    if (head == "conv2d") return parse_conv2d_layer_spec(tokens, spec);
    if (head == "pool1d") return parse_pool1d_layer_spec(tokens, spec);
    if (head == "pool2d") return parse_pool2d_layer_spec(tokens, spec);
    if (head == "residual" || head == "residual_block")
        return parse_residual_layer_spec(tokens, spec);
    if (head == "lstm") return parse_lstm_layer_spec(tokens, spec);

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

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_AUTOENCODER_LAYER_SPEC_PARSERS_HPP
