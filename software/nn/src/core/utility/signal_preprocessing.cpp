#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utility/SignalPreprocessing.hpp"

namespace nn::utility
{

auto read_csv_signal(const std::filesystem::path& path) -> nn::Tensor
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        throw std::runtime_error("Cannot open CSV signal file: " + path.string());
    }

    // Every comma-separated field must be a number. An empty or non-numeric field is an
    // error naming the line and field (they used to be skipped, so a header or a missing
    // value silently shifted every later sample by one).
    std::vector<float> values;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(in, line))
    {
        ++line_number;
        std::stringstream ss(line);
        std::string token;
        std::size_t field = 0;
        while (std::getline(ss, token, ','))
        {
            ++field;
            std::size_t used = 0;
            float value = 0.0F;
            try
            {
                value = std::stof(token, &used);
            }
            catch (const std::exception&)
            {
                used = 0;
            }
            if (token.empty() || used != token.size())
            {
                throw std::runtime_error(
                    "CSV signal " + path.string() + ", line " + std::to_string(line_number) +
                    ", field " + std::to_string(field) + ": '" + token + "' is not a number");
            }
            values.push_back(value);
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
    for (nn::Index i = 0; i < signal.size(); ++i)
    {
        const float d = signal.at(i) - mean;
        sq_sum += d * d;
    }

    const float variance = sq_sum / static_cast<float>(signal.size());
    const float stddev = std::sqrt(std::max(variance, 1e-8F));

    for (nn::Index i = 0; i < signal.size(); ++i) signal.at(i) = (signal.at(i) - mean) / stddev;
}

} // namespace nn::utility
