#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/utility/SignalPreprocessing.hpp"

namespace nn::utility
{

auto read_csv_signal(const std::filesystem::path& path) -> nn::Tensor
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        throw std::runtime_error("Cannot open CSV signal file: " + path.string());
    }

    std::vector<float> values;
    std::string line;
    while (std::getline(in, line))
    {
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            if (token.empty()) continue;
            try
            {
                values.push_back(std::stof(token));
            }
            catch (const std::exception&)
            {
                // Ignore non-numeric tokens by design.
            }
        }
    }

    nn::Tensor signal(static_cast<nn::Index>(values.size()), 1);
    for (nn::Index i = 0; i < signal.rows(); ++i)
    {
        signal.at(i, 0) = values[static_cast<std::size_t>(i)];
    }

    return signal;
}

void zscore_inplace(nn::Tensor& signal)
{
    if (signal.size() == 0) return;

    const float mean = signal.sum() / static_cast<float>(signal.size());

    float sq_sum = 0.0F;
    for (nn::Index i = 0; i < signal.rows(); ++i)
    {
        for (nn::Index j = 0; j < signal.cols(); ++j)
        {
            const float d = signal.at(i, j) - mean;
            sq_sum += d * d;
        }
    }

    const float variance = sq_sum / static_cast<float>(signal.size());
    const float stddev = std::sqrt(std::max(variance, 1e-8F));

    for (nn::Index i = 0; i < signal.rows(); ++i)
    {
        for (nn::Index j = 0; j < signal.cols(); ++j)
        {
            signal.at(i, j) = (signal.at(i, j) - mean) / stddev;
        }
    }
}

} // namespace nn::utility
