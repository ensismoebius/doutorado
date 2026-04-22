#ifndef NN_DATALOADERS_DATALOADER_HPP
#define NN_DATALOADERS_DATALOADER_HPP

#include <cstddef>
#include <memory>
#include <optional>

#include "nn/dataLoaders/datasets/Dataset.hpp"
#include "nn/dataLoaders/runtime/DataLoaderIterator.hpp"
#include "nn/dataLoaders/runtime/DefaultSamplerType.hpp"
#include "nn/dataLoaders/samplers/ISampler.hpp"

class DataLoader
{
   public:
     using DefaultSamplerType = nn::dataLoaders::DefaultSamplerType;
     using DefaultSamplerOptions = nn::dataLoaders::DefaultSamplerOptions;

    DataLoader(std::shared_ptr<Dataset> dataset,
        std::size_t batch_size,
        bool do_shuffle = true,
        std::optional<unsigned int> seed = std::nullopt);

    DataLoader(std::shared_ptr<Dataset> dataset,
        std::size_t batch_size,
        const DefaultSamplerOptions& options);

    DataLoader(std::shared_ptr<Dataset> dataset,
        std::size_t batch_size,
        std::unique_ptr<ISampler> sampler);

    using Iterator = DataLoaderIterator;

    [[nodiscard]] auto begin() -> Iterator;
    [[nodiscard]] auto end() -> Iterator;

   private:
    friend class DataLoaderIterator;

    std::shared_ptr<Dataset> dataset_;
    std::size_t batch_size_;
    std::unique_ptr<ISampler> sampler_;
    std::size_t num_batches_;
    mutable std::size_t epoch_ = 0;
};

#endif // NN_DATALOADERS_DATALOADER_HPP