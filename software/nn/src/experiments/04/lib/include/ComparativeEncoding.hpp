#pragma once

#include <cstdint>
#include <string>

#include "nn/tensor/Tensor.hpp"

namespace comparative_autoencoder_experiment
{

using Tensor = nn::Tensor;

auto encode_sample(const Tensor& sample, const std::string& encoding, std::uint32_t seed) -> Tensor;

auto flatten_time_series(const Tensor& sample) -> Tensor;
auto unflatten_time_series(const Tensor& flat, nn::Index rows, nn::Index cols) -> Tensor;

auto apply_snn_architecture_transform(
    const Tensor& encoded, const std::string& architecture, float alpha, float v_th) -> Tensor;

} // namespace comparative_autoencoder_experiment
