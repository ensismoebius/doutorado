/**
 * @file phase00_features.cpp
 * @brief PHASE 0 feature extraction implementation.
 *
 * Current baseline:
 * - wavelet packet energies per channel
 * - optional paraconsistent metrics derived from normalized feature vectors
 */

#include "phase00_features.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "nn/paraconsistent/paraconsistent.h"
#include "nn/wavelet/Types.h"
#include "nn/wavelet/waveletOperations.h"

namespace phase00
{

auto extract_wavelet_features_single_trial(
    const nn::Tensor& signal_data, double duration_sec, int overlap_percent, int sampling_rate)
    -> std::vector<double>
{
    static_cast<void>(duration_sec);
    static_cast<void>(overlap_percent);
    static_cast<void>(sampling_rate);

    constexpr auto lowpass = wavelets::get_wavelet<wavelets::Daub4>();

    std::vector<double> all_channel_energies;
    for (int channel_row_idx = 0; channel_row_idx < static_cast<int>(signal_data.rows());
        ++channel_row_idx)
    {
        nn::Tensor channel_tensor = signal_data.row(channel_row_idx);
        std::vector<double> sig;
        sig.reserve(static_cast<size_t>(channel_tensor.size()));
        for (size_t j = 0; j < channel_tensor.size(); ++j)
        {
            sig.push_back(channel_tensor.at(0, j));
        }

        auto wtr = wavelets::malat(sig, lowpass, wavelets::PACKET_WAVELET, 4); // level 4

        auto approx = wtr.get_wavelet_transforms(0);
        double energy_approx =
            std::inner_product(approx.begin(), approx.end(), approx.begin(), 0.0);
        all_channel_energies.push_back(energy_approx);

        for (int d = 1; d <= wtr.levelsOfTransformation; ++d)
        {
            auto detail = wtr.get_wavelet_transforms(d);
            double energy_detail =
                std::inner_product(detail.begin(), detail.end(), detail.begin(), 0.0);
            all_channel_energies.push_back(energy_detail);
        }
    }

    return all_channel_energies;
}

auto normalize_features(
    std::vector<std::vector<double>>& features, const std::vector<double>& range) -> void
{
    if (features.empty())
    {
        return;
    }

    size_t num_features = features[0].size();
    std::vector<double> min_vals(num_features, std::numeric_limits<double>::max());
    std::vector<double> max_vals(num_features, std::numeric_limits<double>::lowest());

    for (const auto& feat : features)
    {
        for (size_t i = 0; i < num_features; ++i)
        {
            min_vals[i] = std::min(min_vals[i], feat[i]);
            max_vals[i] = std::max(max_vals[i], feat[i]);
        }
    }

    for (auto& feat : features)
    {
        for (size_t i = 0; i < num_features; ++i)
        {
            if (max_vals[i] != min_vals[i])
            {
                feat[i] = range[0] + (((feat[i] - min_vals[i]) / (max_vals[i] - min_vals[i])) *
                                         (range[1] - range[0]));
            }
            else
            {
                feat[i] = range[0];
            }
        }
    }
}

auto verify_normalization(
    const std::vector<std::vector<double>>& features, const std::vector<double>& range) -> bool
{
    if (features.empty())
    {
        return false;
    }

    double min_val = std::numeric_limits<double>::infinity();
    double max_val = -std::numeric_limits<double>::infinity();

    for (const auto& feat : features)
    {
        for (double v : feat)
        {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
        }
    }

    constexpr double eps = 1e-5;
    return min_val >= range[0] - eps && max_val <= range[1] + eps;
}

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels) -> std::tuple<double, double, double, double>
{
    std::map<std::string, std::vector<std::vector<double>>> arrClasses;

    for (size_t i = 0; i < features.size(); ++i)
    {
        std::string class_name = std::to_string(labels[i]);
        arrClasses[class_name].push_back(features[i]);
    }

    unsigned int amountOfClasses = arrClasses.size();
    if (amountOfClasses == 0)
    {
        return {0.0, 0.0, 0.0, 0.0};
    }

    unsigned int featureVectorsPerClass = arrClasses.begin()->second.size();
    unsigned int featureVectorSize = features[0].size();

    normalize_class_feature_vectors(
        amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);

    double alpha =
        calculate_alpha(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double beta =
        calculate_beta(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);

    double g1 = calculate_certainty_degree_g1(alpha, beta);
    double g2 = calculate_contradiction_degree_g2(alpha, beta);

    return {alpha, beta, g1, g2};
}

auto build_label_index(const std::vector<int>& raw_labels)
    -> std::pair<std::vector<int>, std::vector<int>>
{
    std::vector<int> uniques = raw_labels;
    std::sort(uniques.begin(), uniques.end());
    uniques.erase(std::unique(uniques.begin(), uniques.end()), uniques.end());

    std::unordered_map<int, int> mapping;
    for (size_t i = 0; i < uniques.size(); ++i)
    {
        mapping[uniques[i]] = static_cast<int>(i);
    }

    std::vector<int> normalized;
    normalized.reserve(raw_labels.size());
    for (int lbl : raw_labels)
    {
        normalized.push_back(mapping.at(lbl));
    }

    return {normalized, uniques};
}

} // namespace phase00
