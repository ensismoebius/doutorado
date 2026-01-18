#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstddef>
#include <vector>

#include "nn/tensor/Tensor.hpp"
#include "nn/utility/batching.hpp"

/**
 * @file Dataset.hpp
 * @brief Minimal dataset interface (PyTorch-like) used by `DataLoader`.
 *
 * Conventions used throughout this codebase:
 * - `get_item(idx)` returns a `Batch` whose `.inputs` and `.targets` are usually
 *   single-sample tensors (1 x features). This keeps `collate()` simple.
 * - `collate(indices)` builds a batch by stacking samples along rows.
 * - Datasets can override `collate()` for efficiency (e.g., slicing contiguous
 *   storage rather than looping per element).
 */

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
            if (size() > 0)
            {
                Batch first_item = get_item(0); // Get first item to determine column sizes
                return {.inputs = nn::Tensor(0, first_item.inputs.cols()),
                        .targets = nn::Tensor(0, first_item.targets.cols())};
            }
            return Batch{}; // Empty dataset, return default Batch
        }

        // Use the first item to determine column sizes
        Batch first = get_item(indices[0]);
        const size_t cols_in = first.inputs.cols();
        const size_t cols_tg = first.targets.cols();

        nn::Tensor inputs(static_cast<size_t>(indices.size()), cols_in);
        nn::Tensor targets(static_cast<size_t>(indices.size()), cols_tg);

        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            Batch b = get_item(indices[i]);
            for (size_t c = 0; c < cols_in; ++c)
            {
                inputs.at(static_cast<size_t>(i), c) = b.inputs.at(0, c);
            }
            for (size_t c = 0; c < cols_tg; ++c)
            {
                targets.at(static_cast<size_t>(i), c) = b.targets.at(0, c);
            }
        }

        return {.inputs = std::move(inputs), .targets = std::move(targets)};
    }

    [[nodiscard]] virtual auto size() const -> std::size_t = 0;
    virtual ~Dataset() = default;
};
#endif // DATASET_HPP