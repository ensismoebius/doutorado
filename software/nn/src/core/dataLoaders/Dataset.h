#pragma once

#include <cstddef>
#include <vector>

#include "../tensor/Tensor.hpp"
#include "../utility/batching.hpp"

// Abstract dataset interface similar to PyTorch's Dataset
class Dataset
{
   public:
    [[nodiscard]] virtual auto get_item(std::size_t idx) const -> Batch = 0;
    [[nodiscard]] virtual auto collate(const std::vector<std::size_t>& indices) const -> Batch
    {
        // Default collate: allocate matrices and fill them in one pass
        if (indices.empty())
        {
            // Return empty tensors but retain column counts from first item
            // if dataset is not empty, otherwise return default Batch{}
            if (size() > 0) {
                Batch first_item = get_item(0); // Get first item to determine column sizes
                return {.inputs = nn::Tensor(0, first_item.inputs.cols()),
                        .targets = nn::Tensor(0, first_item.targets.cols())};
            }
            return Batch{}; // Empty dataset, return default Batch
        }

        // Use the first item to determine column sizes
        Batch first = get_item(indices[0]);
        const Eigen::Index cols_in = first.inputs.get_data_ref().cols();
        const Eigen::Index cols_tg = first.targets.get_data_ref().cols();

        Eigen::MatrixXf inputs_mat(static_cast<int>(indices.size()), static_cast<int>(cols_in));
        Eigen::MatrixXf targets_mat(static_cast<int>(indices.size()), static_cast<int>(cols_tg));

        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            Batch b = get_item(indices[i]);
            inputs_mat.row(static_cast<int>(i)) = b.inputs.get_data_ref().row(0);
            targets_mat.row(static_cast<int>(i)) = b.targets.get_data_ref().row(0);
        }

        return {.inputs = nn::Tensor(inputs_mat), .targets = nn::Tensor(targets_mat)};
    }

    [[nodiscard]] virtual auto size() const -> std::size_t = 0;
    virtual ~Dataset() = default;
};
