#include <span>
#include <string>

#include "../tensor/Tensor.hpp"
#include "cnpy.h"

struct NnSaver
{
    static auto save_weights(const std::string& prefix, nn::Tensor& weights, nn::Tensor& bias)
        -> void
    {
        cnpy::npy_save(prefix + "_weights.npy",
                       weights.data_ptr(),
                       {static_cast<size_t>(weights.rows()), static_cast<size_t>(weights.cols())},
                       "w");
        cnpy::npy_save(
            prefix + "_bias.npy", bias.data_ptr(), {static_cast<size_t>(bias.size())}, "w");
    }

    static auto load_weights(const std::string& prefix, nn::Tensor& weights, nn::Tensor& bias)
        -> void
    {
        auto loadedWeights = cnpy::npy_load(prefix + "_weights.npy");
        auto loadedBias = cnpy::npy_load(prefix + "_bias.npy");

        // Convert an array pointer to an iterable
        std::span<const float> const b_span(loadedBias.data<float>(), loadedBias.num_vals);
        std::span<const float> const w_span(loadedWeights.data<float>(), loadedWeights.num_vals);

        // Reconstruct the bias and the weights
        bias = nn::Tensor(loadedBias.shape[0], 1);
        weights = nn::Tensor(loadedWeights.shape[0], loadedWeights.shape[1]);

        for (nn::Index i = 0; i < bias.size(); ++i) [[likely]]
        {
            bias.at(i, 0) = b_span[i];
        }

        for (nn::Index i = 0; i < weights.rows(); ++i) [[likely]]
        {
            for (nn::Index j = 0; j < weights.cols(); ++j)
            {
                weights.at(i, j) = w_span[i * weights.cols() + j];
            }
        }
    }
};