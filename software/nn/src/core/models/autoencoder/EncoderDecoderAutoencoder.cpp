/**
 * @file src/experiments/autoencoderRunner/lib/src/autoencoder/EncoderDecoderAutoencoder.cpp
 * @brief Shared implementation for the "two Sequentials" autoencoders.
 */

#include "EncoderDecoderAutoencoder.hpp"

#include <utility>

#include "AutoencoderBuilders.hpp"

EncoderDecoderAutoencoder::EncoderDecoderAutoencoder(nn::Sequential encoder, nn::Sequential decoder)
    : encoder_(std::move(encoder)), decoder_(std::move(decoder))
{
}

auto EncoderDecoderAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    return encoder_.forward(input, requires_grad);
}

auto EncoderDecoderAutoencoder::decode(const Tensor& latent, bool requires_grad) -> Tensor
{
    return decoder_.forward(latent, requires_grad);
}

auto EncoderDecoderAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto EncoderDecoderAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    // Backward runs the halves in reverse: the decoder's input gradient is
    // the encoder's output gradient.
    Tensor grad = decoder_.backward(grad_output);
    return encoder_.backward(grad);
}

auto EncoderDecoderAutoencoder::params() -> std::span<Tensor*>
{
    return collect_params(param_ptrs_, encoder_, decoder_);
}

void EncoderDecoderAutoencoder::reset_state()
{
    autoencoderRunner::autoencoders::reset_sequential_state(encoder_);
    autoencoderRunner::autoencoders::reset_sequential_state(decoder_);
}
