#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../include/ComparativeCli.hpp"
#include "../include/ComparativeDataset.hpp"
#include "../include/ComparativeOutput.hpp"
#include "../include/ComparativeTraining.hpp"
#include "../include/Experiment04Cli.hpp"
#include "nn/progress/ProgressManager.hpp"
#include "nn/utility/progress.hpp"

// Helper to extract sizes from layer specs
auto extract_layer_sizes(const std::vector<std::string>& specs)
{
    std::vector<int> sizes;
    for (const auto& spec : specs)
    {
        std::stringstream ss(spec);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ':'))
        {
            parts.push_back(token);
        }
        if (parts.size() >= 2 && parts[0] == "linear")
        {
            try
            {
                sizes.push_back(std::stoi(parts[1]));
            }
            catch (...)
            {
            }
        }
    }
    return sizes;
}

namespace lstm_autoencoder_experiment
{

auto run_comparative_experiment(int argc, char* argv[]) -> int
{
    using namespace comparative_autoencoder_experiment;

    try
    {
        const CliOptions cli = parse_cli(argc, argv);
        if (cli.help)
        {
            print_usage(argv[0]);
            return 0;
        }

        const ComparativeConfig config = load_config(resolve_profile_path(cli), cli);
        const std::size_t cfg_hash = config_hash(config);

        std::filesystem::path out_dir = config.dataset.results_dir.empty()
                                            ? source_results_dir()
                                            : std::filesystem::path(config.dataset.results_dir);

        if (!std::filesystem::exists(out_dir))
        {
            out_dir = std::filesystem::path("results");
        }

        std::filesystem::create_directories(out_dir);

        std::vector<ResultRow> all_rows;

        // Total outer iterations: datasets × encodings × repeats.
        const int total_outer_runs = static_cast<int>(config.evaluation.datasets.size()) *
                                     static_cast<int>(config.evaluation.encodings.size()) *
                                     config.experiment.repeats;

        const uint32_t run_bar = nn::progress::ProgressManager::instance().create_bar(
            "all exp. runs", static_cast<float>(total_outer_runs));

        int completed_runs = 0;

        for (const auto& dataset_name : config.evaluation.datasets)
        {
            const DatasetSplit split = build_split(config, dataset_name);

            for (const auto& encoding : config.evaluation.encodings)
            {
                for (int run_id = 0; run_id < config.experiment.repeats; ++run_id)
                {
                    nn::progress::ProgressManager::instance().update_bar(run_bar,
                        static_cast<float>(completed_runs),
                        {{"ds", static_cast<float>(config.evaluation.datasets.size())}});
                    const std::uint32_t run_seed =
                        config.experiment.seed_deterministic
                            ? config.experiment.seed
                            : config.experiment.seed + static_cast<std::uint32_t>(run_id);

                    {
                        float train_ms = 0.0f;
                        float infer_ms = 0.0f;
                        auto lstm_cfg = make_lstm_cfg(config);

                        nn::models::lstm::LSTMAutoencoder lstm_model(lstm_cfg);

                        Adam lstm_opt(config.training.learning_rate);
                        lstm_opt.attach(lstm_model.params());

                        RunMetrics metrics = train_with_early_stopping_lstm(     //
                            lstm_model,                                          //
                            lstm_opt,                                            //
                            config,                                              //
                            split.train_samples,                                 //
                            split.val_samples,                                   //
                            encoding,                                            //
                            run_seed,                                            //
                            static_cast<std::size_t>(run_id),                    //
                            static_cast<std::size_t>(config.experiment.repeats), //
                            train_ms,                                            //
                            infer_ms                                             //
                        );
                        metrics.train_ms = train_ms;

                        all_rows.push_back( //
                            ResultRow{
                                dataset_name, //
                                "lstm-ae",    //
                                encoding,     //
                                "lstm",       //
                                1,            //
                                0.0f,         //
                                0.0f,         //
                                run_id + 1,   //
                                run_seed,     //
                                cfg_hash,     //
                                metrics       //
                            } //
                        );
                    }

                    for (const auto& architecture : config.evaluation.snn_architectures)
                    {
                        for (float voltage_threshold : config.evaluation.v_th_values)
                        {
                            for (float alpha : config.evaluation.alpha_values)
                            {
                                float train_ms = 0.0f;
                                float infer_ms = 0.0f;

                                AutoencoderConfig snn_config = make_snn_cfg( //
                                    config,                                  //
                                    alpha,                                   //
                                    voltage_threshold                        //
                                );
                                snn_config.initializer_seed = run_seed;
                                snn_config.initializer_sampler_type =
                                    "comparative|" + dataset_name + "|" + encoding + "|" +
                                    architecture + "|" +
                                    std::to_string(
                                        extract_layer_sizes(config.model.encoder_layer_spec).empty()
                                            ? 0
                                            : extract_layer_sizes(config.model.encoder_layer_spec)
                                                  .front()) +
                                    "|" + std::to_string(voltage_threshold) + "|" +
                                    std::to_string(alpha);

                                Adam snn_optimizer(config.training.learning_rate);

                                ProtocolSpikingAutoencoder snn_model(snn_config);
                                snn_optimizer.attach(snn_model.params());

                                RunMetrics metrics = train_with_early_stopping_snn(      //
                                    snn_model,                                           //
                                    snn_optimizer,                                       //
                                    config,                                              //
                                    split.train_samples,                                 //
                                    split.val_samples,                                   //
                                    split.val_labels,                                    //
                                    encoding,                                            //
                                    architecture,                                        //
                                    alpha,                                               //
                                    voltage_threshold,                                   //
                                    run_seed,                                            //
                                    static_cast<std::size_t>(run_id),                    //
                                    static_cast<std::size_t>(config.experiment.repeats), //
                                    train_ms,                                            //
                                    infer_ms                                             //
                                );

                                metrics.train_ms = train_ms;
                                all_rows.push_back( //
                                    ResultRow{
                                        dataset_name,                                             //
                                        "snn-ae",                                                 //
                                        encoding,                                                 //
                                        architecture,                                             //
                                        static_cast<int>(config.model.encoder_layer_spec.size()), //
                                        voltage_threshold,                                        //
                                        alpha,                                                    //
                                        run_id + 1,                                               //
                                        run_seed,                                                 //
                                        cfg_hash,                                                 //
                                        metrics                                                   //
                                    } //
                                );
                            }
                        }
                    }

                    nn::progress::ProgressManager::instance().update_bar(
                        run_bar, static_cast<float>(++completed_runs));
                }
            }
        }

        nn::progress::ProgressManager::instance().complete_bar(run_bar);

        if (config.experiment.repeats > 1 && config.experiment.check_determinism)
        {
            validate_repeat_determinism(config, all_rows);
        }

        const std::filesystem::path csv_path =
            out_dir / (config.experiment.run_tag + "_comparative_metrics.csv");

        write_rows_csv(csv_path, all_rows);

        const std::filesystem::path table_path =
            out_dir / (config.experiment.run_tag + "_publication_table.csv");
        write_publication_table(table_path, all_rows);

        const std::filesystem::path summary_json =
            out_dir / (config.experiment.run_tag + "_summary.json");
        write_summary_json(summary_json, config, cfg_hash, all_rows);

        std::cout << "[comparative] Results written to:\n"
                  << "  - " << csv_path << "\n"
                  << "  - " << table_path << "\n"
                  << "  - " << summary_json << "\n";

        flushProgressAsync();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[comparative] Fatal error: " << ex.what() << "\n";
        return 1;
    }
}

} // namespace lstm_autoencoder_experiment
