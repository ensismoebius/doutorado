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

        const ComparativeConfig cfg = load_config(resolve_profile_path(cli));
        const std::size_t cfg_hash = config_hash(cfg);

        std::filesystem::path out_dir =
            cfg.results_dir.empty() ? source_results_dir() : std::filesystem::path(cfg.results_dir);
        if (!std::filesystem::exists(out_dir)) out_dir = std::filesystem::path("results");
        std::filesystem::create_directories(out_dir);

        std::vector<ResultRow> all_rows;

        for (const auto& dataset : cfg.datasets)
        {
            const DatasetSplit split = build_split(cfg, dataset);

            for (const auto& encoding : cfg.encodings)
            {
                for (int run_id = 0; run_id < cfg.repeats; ++run_id)
                {
                    const std::uint32_t run_seed = cfg.seed;

                    {
                        auto lstm_cfg = make_lstm_cfg(cfg);
                        nn::models::lstm::LSTMAutoencoder lstm_model(lstm_cfg);
                        Adam lstm_opt(cfg.learning_rate);
                        lstm_opt.attach(lstm_model.params());

                        float train_ms = 0.0f;
                        float infer_ms = 0.0f;
                        RunMetrics metrics = train_with_early_stopping_lstm(lstm_model,
                            lstm_opt,
                            cfg,
                            split.train_samples,
                            split.val_samples,
                            encoding,
                            run_seed,
                            train_ms,
                            infer_ms);
                        metrics.train_ms = train_ms;

                        all_rows.push_back(ResultRow{dataset,
                            "lstm-ae",
                            encoding,
                            "lstm",
                            1,
                            0.0f,
                            0.0f,
                            run_id + 1,
                            run_seed,
                            cfg_hash,
                            metrics});
                    }

                    for (const auto& architecture : cfg.snn_architectures)
                    {
                        for (int layers : cfg.layers)
                        {
                            for (float v_th : cfg.v_th_values)
                            {
                                for (float alpha : cfg.alpha_values)
                                {
                                    AutoencoderConfig snn_cfg =
                                        make_snn_cfg(cfg, layers, alpha, v_th);

                                    ProtocolSpikingAutoencoder snn_model(snn_cfg);
                                    Adam snn_opt(cfg.learning_rate);
                                    snn_opt.attach(snn_model.params());

                                    float train_ms = 0.0f;
                                    float infer_ms = 0.0f;
                                    RunMetrics metrics = train_with_early_stopping_snn(snn_model,
                                        snn_opt,
                                        cfg,
                                        split.train_samples,
                                        split.val_samples,
                                        split.val_labels,
                                        encoding,
                                        architecture,
                                        layers,
                                        alpha,
                                        v_th,
                                        run_seed,
                                        train_ms,
                                        infer_ms);
                                    metrics.train_ms = train_ms;

                                    all_rows.push_back(ResultRow{dataset,
                                        "snn-ae",
                                        encoding,
                                        architecture,
                                        layers,
                                        v_th,
                                        alpha,
                                        run_id + 1,
                                        run_seed,
                                        cfg_hash,
                                        metrics});
                                }
                            }
                        }
                    }
                }
            }
        }

        validate_repeat_determinism(cfg, all_rows);

        const std::filesystem::path csv_path = out_dir / (cfg.run_tag + "_comparative_metrics.csv");
        write_rows_csv(csv_path, all_rows);

        const std::filesystem::path table_path = out_dir / (cfg.run_tag + "_publication_table.csv");
        write_publication_table(table_path, all_rows);

        const std::filesystem::path summary_json = out_dir / (cfg.run_tag + "_summary.json");
        write_summary_json(summary_json, cfg, cfg_hash, all_rows);

        std::cout << "[comparative] Results written to:\n"
                  << "  - " << csv_path << "\n"
                  << "  - " << table_path << "\n"
                  << "  - " << summary_json << "\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "[comparative] Fatal error: " << ex.what() << "\n";
        return 1;
    }
}

} // namespace lstm_autoencoder_experiment
