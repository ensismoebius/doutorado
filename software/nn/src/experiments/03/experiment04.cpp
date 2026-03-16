/**
 * @file experiment04.cpp
 * @brief PyTorch-style loading/feeding demo for the 10.1117 EEG+Audio dataset.
 *
 * Pipeline implemented here:
 *   Subject directories (S01, S02, ...) -> Dataset -> DataLoader -> Batch -> Model
 */

#include <cstddef>
#include <iostream>
#include <memory>

#include "lib/include/DemoProbeModel.hpp"
#include "lib/include/cli.hpp"
#include "nn/dataLoaders/10.1117/BatchTargetFormatter.hpp"
#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"

using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;
using std::size_t;

auto main(int argc, char* argv[]) -> int
{
    const Config default_config{
        .subject_regex_pattern = "^S(\\d+)$",
        .dataset_root =
            "/home/ensismoebius/Documentos"
            "/UNESP/doutorado/databases/"
            "BaseDeDatosHablaImaginada",
        .batch_size = 4,
        .max_batches = 200,
        .shuffle = true,
        .seed = 42U,
        .sampler_type = "",
        .sampler_weights = {},
        .weighted_num_samples = std::nullopt,
        .distributed_num_replicas = 1,
        .distributed_rank = 0,
        .distributed_shuffle = true,
        .distributed_drop_last = false,
        .input_mode = Protocol101117InputMode::Concatenated,
    };

    Config config = parseCliParams(argc, argv, default_config);

    try
    {
        const auto discovered = discoverSubjects(config.dataset_root, config.subject_regex_pattern);
        const auto dataset = make_shared<Protocol101117Dataset>(discovered);
        dataset->set_input_mode(config.input_mode);

        DataLoader loader(dataset, config.batch_size, config.sampler_options);
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

            cout << nn::dataLoaders::formatProtocol101117BatchTargets(batch);
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
