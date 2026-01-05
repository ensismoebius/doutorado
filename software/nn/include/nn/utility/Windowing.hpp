#ifndef NN_CORE_UTILITY_WINDOWING_HPP
#define NN_CORE_UTILITY_WINDOWING_HPP

#include <vector>

#include "nn/tensor/Tensor.hpp"

class Windowing
{
   public:
    static auto slidingWindow(const nn::Tensor& data, float window_size_sec, float overlap_ratio,
                              int sampling_rate) -> std::vector<nn::Tensor>
    {
        std::vector<nn::Tensor> windows;

        if (data.rows() == 0 || data.cols() == 0)
        {
            return windows;
        }

        int window_size_samples =
            static_cast<int>(window_size_sec * static_cast<double>(sampling_rate));
        int overlap_samples =
            static_cast<int>(static_cast<double>(window_size_samples) * overlap_ratio);
        int step_size_samples = window_size_samples - overlap_samples;

        if (window_size_samples <= 0 || step_size_samples <= 0)
        {
            // Handle invalid window/step sizes
            return windows;
        }

        for (auto i = 0; i + window_size_samples <= static_cast<int>(data.cols());
             i += step_size_samples) [[likely]]
        {
            // Create a new tensor for the window
            nn::Tensor window(static_cast<int>(data.rows()), window_size_samples);
            for (int row = 0; row < static_cast<int>(data.rows()); ++row)
            {
                for (int col = 0; col < window_size_samples; ++col)
                {
                    window(row, col) = data(row, i + col);
                }
            }
            windows.emplace_back(window);
        }

        return windows;
    }
};

#endif // NN_CORE_UTILITY_WINDOWING_HPP
