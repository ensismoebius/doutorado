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
#include "nn/utility/progress.hpp"

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

        std::filesystem::path out_dir = config.results_dir.empty()
                                            ? source_results_dir()
                                            : std::filesystem::path(config.results_dir);

        if (!std::filesystem::exists(out_dir))
        {
            out_dir = std::filesystem::path("results");
        }

        std::filesystem::create_directories(out_dir);

        std::vector<ResultRow> all_rows;

        for (const auto& dataset_name : config.datasets)
        {
            const DatasetSplit split = build_split(config, dataset_name);

            for (const auto& encoding : config.encodings)
            {
                for (int run_id = 0; run_id < config.repeats; ++run_id)
                {
                    const std::uint32_t run_seed =
                        config.seed_deterministic
                            ? config.seed
                            : config.seed + static_cast<std::uint32_t>(run_id);

                    {
                        float train_ms = 0.0f;
                        float infer_ms = 0.0f;
                        auto lstm_cfg = make_lstm_cfg(config);

                        nn::models::lstm::LSTMAutoencoder lstm_model(lstm_cfg);

                        Adam lstm_opt(config.learning_rate);
                        lstm_opt.attach(lstm_model.params());

                        RunMetrics metrics = train_with_early_stopping_lstm( //
                            lstm_model,                                      //
                            lstm_opt,                                        //
                            config,                                          //
                            split.train_samples,                             //
                            split.val_samples,                               //
                            encoding,                                        //
                            run_seed,                                        //
                            static_cast<std::size_t>(run_id),                //
                            static_cast<std::size_t>(config.repeats),        //
                            train_ms,                                        //
                            infer_ms                                         //
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

                    for (const auto& architecture : config.snn_architectures)
                    {
                        for (float voltage_threshold : config.v_th_values)
                        {
                            for (float alpha : config.alpha_values)
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
                                    architecture + "|" + std::to_string(config.layer_sizes.empty() ? 0 : config.layer_sizes.front()) + "|" +
                                    std::to_string(voltage_threshold) + "|" +
                                    std::to_string(alpha);

                                Adam snn_optimizer(config.learning_rate);

                                ProtocolSpikingAutoencoder snn_model(snn_config);
                                snn_optimizer.attach(snn_model.params());

                                RunMetrics metrics = train_with_early_stopping_snn( //
                                    snn_model,                                      //
                                    snn_optimizer,                                  //
                                    config,                                         //
                                    split.train_samples,                            //
                                    split.val_samples,                              //
                                    split.val_labels,                               //
                                    encoding,                                       //
                                    architecture,                                   //
                                    alpha,                                          //
                                    voltage_threshold,                              //
                                    run_seed,                                       //
                                    static_cast<std::size_t>(run_id),               //
                                    static_cast<std::size_t>(config.repeats),       //
                                    train_ms,                                       //
                                    infer_ms                                        //
                                );

                                metrics.train_ms = train_ms;
                                all_rows.push_back( //
                                    ResultRow{
                                        dataset_name,      //
                                        "snn-ae",          //
                                        encoding,          //
                                        architecture,      //
                                        static_cast<int>(config.layer_sizes.size()),        //
                                        voltage_threshold, //
                                        alpha,             //
                                        run_id + 1,        //
                                        run_seed,          //
                                        cfg_hash,          //
                                        metrics            //
                                    } //
                                );
                            }
                        }
                    }
                }
            }
        }

        if (config.repeats > 1 && config.check_determinism)
        {
            validate_repeat_determinism(config, all_rows);
        }

        const std::filesystem::path csv_path =
            out_dir / (config.run_tag + "_comparative_metrics.csv");

        write_rows_csv(csv_path, all_rows);

        const std::filesystem::path table_path =
            out_dir / (config.run_tag + "_publication_table.csv");
        write_publication_table(table_path, all_rows);

        const std::filesystem::path summary_json = out_dir / (config.run_tag + "_summary.json");
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
