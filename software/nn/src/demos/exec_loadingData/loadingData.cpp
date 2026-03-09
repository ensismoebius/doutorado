/**
 * @file loadingData.cpp
 * @brief PyTorch-style loading/feeding demo for the 10.1117 EEG+Audio dataset.
 *
 * Pipeline implemented here:
 *   Subject directories (S01, S02, ...) -> Dataset -> DataLoader -> Batch -> Model
 */

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "lib/include/BatchPrefetcher.hpp"
#include "lib/include/DemoProbeModel.hpp"
#include "lib/include/Protocol101117Dataset.hpp"
#include "lib/include/batch_util.hpp"
#include "lib/include/cli.hpp"
#include "lib/include/subject_discovery.hpp"
#include "nn/dataLoaders/DataLoader.hpp"

using std::cerr;
using std::cout;
using std::exception;
using std::size_t;

namespace
{

auto makeSamplerOptions(const Config& config) -> DataLoader::DefaultSamplerOptions
{
    DataLoader::DefaultSamplerOptions options{};
    options.seed = config.seed;

    if (config.sampler_type.empty())
    {
        options.type = config.shuffle ? DataLoader::DefaultSamplerType::Random
                                      : DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "sequential")
    {
        options.type = DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "random")
    {
        options.type = DataLoader::DefaultSamplerType::Random;
        return options;
    }

    if (config.sampler_type == "weighted")
    {
        options.type = DataLoader::DefaultSamplerType::WeightedRandom;
        options.weights = config.sampler_weights;
        options.weighted_num_samples = config.weighted_num_samples;
        return options;
    }

    if (config.sampler_type == "distributed")
    {
        options.type = DataLoader::DefaultSamplerType::Distributed;
        options.num_replicas = config.distributed_num_replicas;
        options.rank = config.distributed_rank;
        options.distributed_shuffle = config.distributed_shuffle;
        options.distributed_drop_last = config.distributed_drop_last;
        return options;
    }

    throw std::runtime_error("Unknown sampler type: " + config.sampler_type);
}

} // namespace

auto main(int argc, char* argv[]) -> int
{
    const Config default_config{
        .subject_regex_pattern = "^S(\\d+)$",
        .dataset_root =
            "/home/ensismoebius/Documentos"
            "/UNESP/doutorado/databases/"
            "BaseDeDatosHablaImaginada",
        .batch_size = 4,
        .max_batches = 10,
        .shuffle = true,
        .seed = 42U,
        .sampler_type = "",
        .sampler_weights = {},
        .weighted_num_samples = std::nullopt,
        .distributed_num_replicas = 1,
        .distributed_rank = 0,
        .distributed_shuffle = true,
        .distributed_drop_last = false,
    };

    Config config{};
    parseCliParams(argc, argv, config, default_config);

    try
    {
        auto discovered = discoverSubjects(config.dataset_root, config.subject_regex_pattern);
        auto dataset = std::make_shared<Protocol101117Dataset>(discovered);

        DataLoader loader(dataset, config.batch_size, makeSamplerOptions(config));
        DemoProbeModel model;

        cout << "Dataset root: " << config.dataset_root << '\n';
        cout << "Subjects discovered: " << dataset->subjects().size() << '\n';
        for (const auto& s : dataset->subjects())
        {
            cout << "  - " << s.subject_name << '\n';
        }
        cout << "Total synchronized samples: " << dataset->size() << "\n\n";

        BatchPrefetcher prefetcher(loader, config.max_batches);

        while (prefetcher.hasNext())
        {
            auto maybe_batch = prefetcher.next();
            if (!maybe_batch.has_value())
            {
                break;
            }

            Batch batch = std::move(maybe_batch.value());
            nn::Tensor probe = model.forward(batch.inputs);
            (void) probe;

            printData(batch);
        }

        if (prefetcher.seenBatches() == 0)
        {
            cout << "No batches produced. Check dataset files and row counts.\n";
        }

        cout << "Pipeline completed: Dataset -> DataLoader -> Batch -> Model\n";
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
