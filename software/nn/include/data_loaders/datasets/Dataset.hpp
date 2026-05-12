#ifndef DATASET_HPP
#define DATASET_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "data_loaders/interfaces/IDatasetPrinter.hpp"
#include "tensor/Tensor.hpp"
#include "utility/batching.hpp"

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

    virtual void collate_into(const std::vector<std::size_t>& indices, Batch& batch) const
    {
        if (indices.empty())
        {
            if (size() > 0)
            {
                Batch first_item = get_item(0);
                batch.inputs.reshape({0, static_cast<nn::Index>(first_item.inputs.cols())});
                batch.targets.reshape({0, static_cast<nn::Index>(first_item.targets.cols())});
            }
            return;
        }

        Batch first = get_item(indices[0]);
        const size_t cols_in = first.inputs.cols();
        const size_t cols_tg = first.targets.cols();

        if (batch.inputs.rows() != static_cast<nn::Index>(indices.size()) ||
            batch.inputs.cols() != static_cast<nn::Index>(cols_in))
        {
            batch.inputs = nn::Tensor(static_cast<nn::Index>(indices.size()), cols_in);
        }
        if (batch.targets.rows() != static_cast<nn::Index>(indices.size()) ||
            batch.targets.cols() != static_cast<nn::Index>(cols_tg))
        {
            batch.targets = nn::Tensor(static_cast<nn::Index>(indices.size()), cols_tg);
        }

        for (std::size_t row = 0; row < indices.size(); ++row)
        {
            Batch b = get_item(indices[row]);
            if (b.inputs.rows() != 1 || b.inputs.cols() != cols_in)
            {
                throw std::invalid_argument("Dataset::collate_into: inconsistent input shape");
            }
            if (b.targets.rows() != 1 || b.targets.cols() != cols_tg)
            {
                throw std::invalid_argument("Dataset::collate_into: inconsistent target shape");
            }

            batch.inputs.setBlock(row, 0, b.inputs);
            batch.targets.setBlock(row, 0, b.targets);
        }
    }

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

        for (std::size_t row = 0; row < indices.size(); ++row)
        {
            Batch b = get_item(indices[row]);

            // Default collate expects single-sample rows from get_item().
            if (b.inputs.rows() != 1 || b.inputs.cols() != cols_in)
            {
                throw std::invalid_argument("Dataset::collate: inconsistent input sample shape");
            }
            if (b.targets.rows() != 1 || b.targets.cols() != cols_tg)
            {
                throw std::invalid_argument("Dataset::collate: inconsistent target sample shape");
            }

            inputs.setBlock(row, 0, b.inputs);
            targets.setBlock(row, 0, b.targets);
        }

        return {.inputs = std::move(inputs), .targets = std::move(targets)};
    }

    /**
     * Print dataset summary using the provided printer strategy.
     *
     * Subclasses should override this to delegate to the appropriate printer method.
     * Default implementation calls `print_generic()` on the printer.
     *
     * @param printer The printer strategy to use for formatting output.
     */
    virtual void print(IDatasetPrinter& printer) const
    {
        printer.print_generic(*this);
    }

    [[nodiscard]] virtual auto size() const -> std::size_t = 0;
    virtual ~Dataset() = default;
};
#endif // DATASET_HPP