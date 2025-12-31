#ifndef NN_CORE_UTILITY_NORMALIZATION_HPP
#define NN_CORE_UTILITY_NORMALIZATION_HPP

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
        if (data.get_data_ref().size() == 0)
        {
            return data;
        }

        auto data_matrix = data.get_data_ref();
        float min_val = data_matrix.minCoeff();
        float max_val = data_matrix.maxCoeff();

        if (max_val == min_val)
        {
            return nn::Tensor(data.get_shape());
        }

        Eigen::MatrixXf normalized_data = (data_matrix.array() - min_val) / (max_val - min_val);
        return nn::Tensor(normalized_data);
    }
};

#endif // NN_CORE_UTILITY_NORMALIZATION_HPP
