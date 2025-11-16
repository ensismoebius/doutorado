#ifndef NN_CORE_UTILITY_WINDOWING_HPP
#define NN_CORE_UTILITY_WINDOWING_HPP

#include <Eigen/Dense>
#include <vector>

class Windowing
{
   public:
    static auto slidingWindow(const Eigen::MatrixXf& data, float window_size_sec,
                              float overlap_ratio, int sampling_rate)
        -> std::vector<Eigen::MatrixXf>
    {
        std::vector<Eigen::MatrixXf> windows;

        if (data.size() == 0)
        {
            return windows;
        }

        int window_size_samples = static_cast<int>(window_size_sec * sampling_rate);
        int overlap_samples = static_cast<int>(window_size_samples * overlap_ratio);
        int step_size_samples = window_size_samples - overlap_samples;

        if (window_size_samples <= 0 || step_size_samples <= 0)
        {
            // Handle invalid window/step sizes
            return windows;
        }

        for (int i = 0; i + window_size_samples <= data.cols(); i += step_size_samples)
        {
            windows.emplace_back(data.block(0, i, data.rows(), window_size_samples));
        }

        return windows;
    }
};

#endif // NN_CORE_UTILITY_WINDOWING_HPP
