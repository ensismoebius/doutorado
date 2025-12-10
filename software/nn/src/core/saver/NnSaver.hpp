#include <span>
#include <string>

#include "cnpy.h"
#include "tensor/Tensor.hpp"

struct NnSaver
{
    static auto save_weights(const std::string& prefix, nn::Tensor& weights, nn::Tensor& bias) -> void
    {
        cnpy::npy_save(
            prefix + "_weights.npy",
            weights.get_data_ref().data(),
            {static_cast<size_t>(weights.data.rows()), static_cast<size_t>(weights.data.cols())},
            "w");
        cnpy::npy_save(
            prefix + "_bias.npy", bias.get_data_ref().data(), {static_cast<size_t>(bias.data.size())}, "w");
    }

    static auto load_weights(const std::string& prefix, nn::Tensor& weights, nn::Tensor& bias) -> void
    {
        auto loadedWeights = cnpy::npy_load(prefix + "_weights.npy");
        auto loadedBias = cnpy::npy_load(prefix + "_bias.npy");

        // Convert an array pointer to an iterable
        std::span<const float> const b_span(loadedBias.data<float>(), loadedBias.num_vals);
        std::span<const float> const w_span(loadedWeights.data<float>(), loadedWeights.num_vals);

        // Reconstruct the bias and the weights
        bias.get_data_ref() = Eigen::VectorXf(loadedBias.shape[0]);
        weights.get_data_ref() = Eigen::MatrixXf(loadedWeights.shape[0], loadedWeights.shape[1]);

        for (int i = 0; i < bias.get_data_ref().size(); ++i)
        {
            bias.get_data_ref()(i) = b_span[i];
        }

        for (int i = 0; i < weights.get_data_ref().size(); ++i)
        {
            weights.get_data_ref()(i) = w_span[i];
        }
    }
};