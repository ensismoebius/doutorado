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
#include "lib/include/dataset_info.hpp"
#include "lib/include/progress.hpp"
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
        .batch_size = 5,
        .max_batches = 1000,
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
        .lookahead = 4,
    };

    Config config = parseCliParams(argc, argv, default_config);

    try
    {
        const auto discovered = discoverSubjects(config.dataset_root, config.subject_regex_pattern);
        const auto dataset = make_shared<Protocol101117Dataset>(discovered);
        dataset->set_input_mode(config.input_mode);

        DataLoader loader(dataset, config.batch_size, config.sampler_options);
        DemoProbeModel model;

        printDatasetSummary(*dataset, config.dataset_root);

        BatchPrefetcher prefetcher(loader, config.max_batches, config.lookahead);

        const size_t dataset_total_samples = dataset->size();
        size_t processed_samples = 0;

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

            // Update processed samples and print progress (in-place)
            processed_samples += batch.inputs.rows();
            const size_t seen_batches = prefetcher.seenBatches();
            printProgress(dataset_total_samples,
                          config.batch_size,
                          config.max_batches,
                          seen_batches,
                          processed_samples,
                          false);

            // cout << nn::dataLoaders::formatProtocol101117BatchTargets(batch);
        }

        // Finalize progress line
        printProgress(dataset_total_samples,
                      config.batch_size,
                      config.max_batches,
                      prefetcher.seenBatches(),
                      processed_samples,
                      true);

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
