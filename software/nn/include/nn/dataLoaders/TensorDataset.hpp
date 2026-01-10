#pragma once

#include <algorithm>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

#include "Dataset.hpp"

/**
 * @file TensorDataset.hpp
 * @brief Dataset backed by in-memory tensors.
 *
 * Shape convention:
 * - `inputs` and `targets` are expected to use the first dimension as the number
 *   of samples (rows in a 2D view).
 * - `get_item(i)` returns a batch with a single row (1 x features) by slicing.
 *
 * Performance note:
 * - `collate(indices)` uses `Tensor::slice` to gather multiple rows in one call.
 *   This is typically more efficient than the elementwise default `Dataset::collate`.
 */

class TensorDataset : public Dataset
{
   public:
    TensorDataset() = default;
    TensorDataset(nn::Tensor inputs, nn::Tensor targets)
        : inputs_(std::move(inputs)), targets_(std::move(targets))
    {
        if (inputs_.get_shape()[0] != targets_.get_shape()[0])
        {
            throw std::invalid_argument(
                "TensorDataset: inputs and targets must have the same number of samples. Got "
                "inputs: " +
                std::to_string(inputs_.get_shape()[0]) +
                ", targets: " + std::to_string(targets_.get_shape()[0]));
        }
    }

    [[nodiscard]] auto get_item(std::size_t idx) const -> Batch override
    {
        std::vector<int> indices{static_cast<int>(idx)};
        nn::Tensor in = inputs_.slice(std::span<const int>(indices));
        nn::Tensor tg = targets_.slice(std::span<const int>(indices));
        return {.inputs = in, .targets = tg};
    }

    auto collate(const std::vector<std::size_t>& indices) const -> Batch override
    {
        std::vector<int> idxs;
        idxs.reserve(indices.size());
        std::transform(indices.begin(),
                       indices.end(),
                       std::back_inserter(idxs),
                       [](std::size_t i) { return static_cast<int>(i); });
        return {.inputs = inputs_.slice(std::span<const int>(idxs)),
                .targets = targets_.slice(std::span<const int>(idxs))};
    }

    [[nodiscard]] auto size() const -> std::size_t override
    {
        return static_cast<std::size_t>(inputs_.get_shape()[0]);
    }

   private:
    nn::Tensor inputs_;
    nn::Tensor targets_;

   protected:
    void set_tensors(nn::Tensor inputs, nn::Tensor targets)
    {
        inputs_ = std::move(inputs);
        targets_ = std::move(targets);
    }
};
