#ifndef NN_CORE_UTILITY_NORMALIZATION_HPP
#define NN_CORE_UTILITY_NORMALIZATION_HPP

#include <Eigen/Dense>

class Normalization
{
   public:
    static auto minMax(const Eigen::MatrixXf& data) -> Eigen::MatrixXf
    {
        if (data.size() == 0)
        {
            return data;
        }
        Eigen::MatrixXf normalized_data = data;
        float min_val = data.minCoeff();
        float max_val = data.maxCoeff();

        if (max_val == min_val)
        {
            return Eigen::MatrixXf::Zero(data.rows(), data.cols());
        }

        normalized_data = (data.array() - min_val) / (max_val - min_val);
        return normalized_data;
    }
};

#endif // NN_CORE_UTILITY_NORMALIZATION_HPP
