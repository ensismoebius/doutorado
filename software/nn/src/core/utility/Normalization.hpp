#ifndef NN_CORE_UTILITY_NORMALIZATION_HPP
#define NN_CORE_UTILITY_NORMALIZATION_HPP

#include <algorithm>

#include "../tensor/Tensor.hpp"

class Normalization
{
   public:
    /**
     * @brief Normalize data to [0, 1] range
     *
     * @param data Input tensor
     * @return nn::Tensor
     */
    static auto normalize_0_1(const nn::Tensor& data) -> nn::Tensor
    {
        if (data.size() == 0)
        {
            return data;
        }

        float min_val = data.at(0, 0);
        float max_val = data.at(0, 0);

        for (nn::Index i = 0; i < data.rows(); ++i)
        {
            for (nn::Index j = 0; j < data.cols(); ++j)
            {
                const float value = data.at(i, j);
                min_val = std::min(min_val, value);
                max_val = std::max(max_val, value);
            }
        }

        if (max_val == min_val)
        {
            return nn::Tensor(data.get_shape());
        }

        nn::Tensor normalized(data.rows(), data.cols());
        const float range = max_val - min_val;

        for (nn::Index i = 0; i < data.rows(); ++i)
        {
            for (nn::Index j = 0; j < data.cols(); ++j)
            {
                normalized.at(i, j) = (data.at(i, j) - min_val) / range;
            }
        }

        return normalized;
    }
};

#endif // NN_CORE_UTILITY_NORMALIZATION_HPP
