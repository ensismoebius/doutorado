#include "experiment03.hpp"

#include <iostream>

#include "DemoProbeModel.hpp"
#include "dataset_info.hpp"
#include "experiment03.hpp"
#include "nn/dataLoaders/10.1117/codec/BatchTargetFormatter.hpp"
#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/10.1117/windowing/AudioWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/EEGWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/FusedWindowDataset.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "progress.hpp"

using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;

Experiment03::Experiment03(const Config& config) : config_(config)
{
    // Initialize the demo probe model (no actual learning,
    // just a forward pass).
    model_ = std::make_unique<DemoProbeModel>();
    // Dataset, DataLoader and BatchPrefetcher will be initialized in `run()`
    // after parsing CLI params and discovering subjects.
}

int Experiment03::run()
{
    try
    {
        // Initialize processed samples count for progress tracking.
        processed_samples_ = 0;

        // Discover subjects and initialize dataset with specified input mode.
        const auto discovered =
            discoverSubjects(config_.dataset_root, config_.subject_regex_pattern);

        // Instantiate dataset based on configured type.
        // Dataset is shared_ptr so it can be easily passed to DataLoader
        // and persist across the experiment.
        switch (config_.dataset_type)
        {
            case Experiment03DatasetType::Protocol:
            {
                auto proto_dataset = make_shared<Protocol101117Dataset>(discovered);
                proto_dataset->set_input_mode(config_.input_mode);
                dataset_ = proto_dataset;
                break;
            }
            case Experiment03DatasetType::EegWindow:
            {
                dataset_ = make_shared<EEGWindowDataset>(discovered, config_.eeg_window_spec);
                break;
            }
            case Experiment03DatasetType::AudioWindow:
            {
                dataset_ = make_shared<AudioWindowDataset>(discovered, config_.audio_window_spec);
                break;
            }
            case Experiment03DatasetType::FusedWindow:
            {
                dataset_ = make_shared<FusedWindowDataset>(
                    discovered, config_.eeg_window_spec, config_.audio_window_spec);
                break;
            }
        }

        // DataLoader is unique_ptr since it owns the iteration
        // state and should not be shared.
        loader_ =
            std::make_unique<DataLoader>(dataset_, config_.batch_size, config_.sampler_options);

        // BatchPrefetcher is unique_ptr since it manages background threads
        // and should not be shared.
        prefetcher_ =
            std::make_unique<BatchPrefetcher>(*loader_, config_.max_batches, config_.lookahead);

        // Store total dataset size for progress tracking.
        dataset_total_samples_ = dataset_->size();

        // Print dataset summary before processing batches.
        // Protocol datasets have a specialized summary; windowing datasets print basic info.
        if (config_.dataset_type == Experiment03DatasetType::Protocol)
        {
            auto* proto = dynamic_cast<Protocol101117Dataset*>(dataset_.get());
            if (proto) printDatasetSummary(*proto, config_.dataset_root);
        }
        else
        {
            cout << "Dataset initialized with " << dataset_total_samples_ << " total samples."
                 << std::endl;
        }

        // Main loop: iterate over batches from the prefetcher, run the model.
        // Print progress is printed in-place.
        while (prefetcher_->hasNext())
        {
            // Get the next batch from the prefetcher;
            // break if no more batches are available.
            auto maybe_batch = prefetcher_->next();
            if (!maybe_batch.has_value()) break;

            // Move the batch out of the optional for processing.
            Batch batch = std::move(maybe_batch.value());

            // Future model inference would go here;
            // for now we just run the forward pass
            nn::Tensor probe = model_->forward(batch.inputs);
            (void) probe;

            // Update processed samples and print progress (in-place)
            processed_samples_ += batch.inputs.rows();
            seen_batches_ = prefetcher_->seenBatches();

            // Print batch targets for debugging. Protocol datasets have a specific
            // formatter; windowing datasets just print batch shape.
            if (config_.dataset_type == Experiment03DatasetType::Protocol)
            {
                cout << nn::dataLoaders::formatProtocol101117BatchTargets(batch);
            }
            else
            {
                cout << "  [Batch shape: inputs=" << batch.inputs.rows() << "x"
                     << batch.inputs.cols() << ", targets=" << batch.targets.rows() << "x"
                     << batch.targets.cols() << "]" << std::endl;
            }
            printProgress(dataset_total_samples_,
                config_.batch_size,
                config_.max_batches,
                seen_batches_,
                processed_samples_,
                false);
        }

        // Finalize progress line
        printProgress(dataset_total_samples_,
            config_.batch_size,
            config_.max_batches,
            prefetcher_->seenBatches(),
            processed_samples_,
            true);

        // Check if any batches were produced; if not, print a warning.
        if (prefetcher_->seenBatches() == 0)
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
