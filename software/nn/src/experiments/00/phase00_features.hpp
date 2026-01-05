#pragma once

#include <tuple>
#include <vector>

#include "nn/tensor/Tensor.hpp"

namespace phase00
{
auto extract_wavelet_features_single_trial(const nn::Tensor& signal_data, double duration_sec,
                                           int overlap_percent, int sampling_rate)
    -> std::vector<double>;

auto normalize_features(std::vector<std::vector<double>>& features,
                        const std::vector<double>& range) -> void;
auto verify_normalization(const std::vector<std::vector<double>>& features,
                          const std::vector<double>& range) -> bool;

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
                                    const std::vector<int>& labels)
    -> std::tuple<double, double, double, double>;

auto build_label_index(const std::vector<int>& raw_labels)
    -> std::pair<std::vector<int>, std::vector<int>>;

} // namespace phase00
