#ifndef NN_DATALOADERS_DATALOADER_HPP
#define NN_DATALOADERS_DATALOADER_HPP

#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <vector>

#include "Dataset.hpp"
#include "nn/dataLoaders/samplers/ISampler.hpp"

/**
 * @file DataLoader.hpp
 * @brief Mini-batch iterator over a `Dataset`.
 *
 * This is intentionally small and "framework-free": it does not manage threads,
 * pinned memory, prefetching, or async I/O. It exists to support the demos and
 * experiments in this repository.
 *
 * Determinism model:
 * - If `seed` is provided, shuffling is deterministic per epoch.
 * - Each iterator captures its own snapshot of the shuffled indices so multiple
 *   iterators can coexist without interfering with each other.
 */

class DataLoader
{
   public:
    enum class DefaultSamplerType
    {
        Sequential,
        Random,
        WeightedRandom,
        Distributed,
    };

    struct DefaultSamplerOptions
    {
        DefaultSamplerType type = DefaultSamplerType::Sequential;

        std::optional<unsigned int> seed = std::nullopt;

        // WeightedRandomSampler options
        std::vector<double> weights = {};
        std::optional<std::size_t> weighted_num_samples = std::nullopt;

        // DistributedSampler options
        std::size_t num_replicas = 1;
        std::size_t rank = 0;
        bool distributed_shuffle = true;
        bool distributed_drop_last = false;
    };

    // Backward-compatible constructor that maps to built-in samplers:
    // - do_shuffle=false: SequentialSampler
    // - do_shuffle=true:  RandomSampler(seed)
    DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size, bool do_shuffle = true,
               std::optional<unsigned int> seed = std::nullopt);

    // Built-in sampler selector constructor.
    DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size,
               DefaultSamplerOptions options);

    // Preferred constructor: inject an arbitrary sampler implementation.
    DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size,
               std::unique_ptr<ISampler> sampler);

    class Iterator
    {
       public:
        using iterator_category = std::input_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Batch;
        using pointer = Batch*;
        using reference = Batch&;

        Iterator(DataLoader& loader, std::size_t current_batch, std::vector<std::size_t> indices);

        auto operator*() const -> Batch;
        auto operator++() -> Iterator&;
        auto operator!=(const Iterator& other) const -> bool;
        auto operator==(const Iterator& other) const -> bool;

       private:
        DataLoader& loader_;
        std::size_t current_batch_;
        std::vector<std::size_t> indices_; // Each iterator has its own snapshot
    };

    [[nodiscard]] auto begin() -> Iterator;
    [[nodiscard]] auto end() -> Iterator;

   private:
    std::shared_ptr<Dataset> dataset_;
    std::size_t batch_size_;
    std::unique_ptr<ISampler> sampler_;
    std::size_t num_batches_;
    // Epoch is propagated to samplers so they can update internal state
    // (e.g. deterministic per-epoch shuffles in RandomSampler).
    mutable std::size_t epoch_ = 0;
};

#endif // NN_DATALOADERS_DATALOADER_HPP
