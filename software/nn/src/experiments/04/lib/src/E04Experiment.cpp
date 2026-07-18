#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>

#include "../include/E04Checkpoint.hpp"
#include "../include/E04Cli.hpp"
#include "../include/E04Dataset.hpp"
#include "../include/E04Output.hpp"
#include "../include/E04Runner.hpp"
#include "../include/E04Training.hpp"
#include "logging/Logger.hpp" // IWYU pragma: keep — provides NN_LOG_* macros
#include "progress/ProgressManager.hpp"
#include "utility/progress.hpp"

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

auto active_backend_name() -> std::string
{
#if defined(NN_BACKEND_OPENCL)
    return "opencl";
#elif defined(NN_BACKEND_DEVICE)
    return "device";
#else
    return "xtensor";
#endif
}

auto sanitize_name(const std::string& raw) -> std::string
{
    std::string out;
    out.reserve(raw.size());
    for (char c : raw)
    {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '-' || c == '_')
        {
            out.push_back(c);
        }
        else
        {
            out.push_back('_');
        }
    }
    return out.empty() ? std::string("artifact") : out;
}

auto save_state_dict_text(
    const std::filesystem::path& path, const std::map<std::string, nn::Tensor>& state_dict) -> bool
{
    std::ofstream out(path);
    if (!out.is_open())
    {
        return false;
    }

    out << "# LSTM state_dict dump (name rows cols values...)\n";
    for (const auto& [name, tensor] : state_dict)
    {
        out << name << ' ' << tensor.rows() << ' ' << tensor.cols();
        for (nn::Index r = 0; r < tensor.rows(); ++r)
        {
            for (nn::Index c = 0; c < tensor.cols(); ++c)
            {
                out << ' ' << tensor.at(r, c);
            }
        }
        out << '\n';
    }

    return out.good();
}

auto save_parameter_list_text(
    const std::filesystem::path& path, std::span<nn::Tensor*> parameters, const std::string& prefix)
    -> bool
{
    std::ofstream out(path);
    if (!out.is_open())
    {
        return false;
    }

    out << "# Parameter dump (name rows cols values...)\n";
    for (std::size_t i = 0; i < parameters.size(); ++i)
    {
        const nn::Tensor* tensor = parameters[i];
        if (tensor == nullptr)
        {
            continue;
        }

        out << prefix << '.' << i << ' ' << tensor->rows() << ' ' << tensor->cols();
        for (nn::Index r = 0; r < tensor->rows(); ++r)
        {
            for (nn::Index c = 0; c < tensor->cols(); ++c)
            {
                out << ' ' << tensor->at(r, c);
            }
        }
        out << '\n';
    }

    return out.good();
}

namespace e04
{

auto run_comparative_experiment(int argc, char* argv[]) -> int
{
    using namespace e04;

    try
    {
        const CliOptions cli = parse_cli(argc, argv);
        if (cli.help)
        {
            print_usage(argv[0]);
            return 0;
        }

        const E04Config config = load_config(resolve_profile_path(cli), cli);
        config.validate();
        const std::size_t cfg_hash = config_hash(config);
        const std::string backend_name = active_backend_name();

        std::filesystem::path out_dir = config.dataset.results_dir.empty()
                                            ? source_results_dir()
                                            : std::filesystem::path(config.dataset.results_dir);

        if (!std::filesystem::exists(out_dir))
        {
            out_dir = std::filesystem::path("results");
        }

        std::filesystem::create_directories(out_dir);
        const std::filesystem::path models_dir =
            out_dir / "models" / sanitize_name(config.experiment.run_tag);
        if (config.dataset.save_models)
        {
            std::filesystem::create_directories(models_dir);
        }

        const auto chk_dir = out_dir / "checkpoints";
        std::filesystem::create_directories(chk_dir);

        std::vector<ResultRow> all_rows;

        // Total individual runs: datasets × encodings × repeats × (1 LSTM + SNN sweep).
        const int snn_per_combo = static_cast<int>(config.evaluation.snn_architectures.size()) *
                                  static_cast<int>(config.evaluation.v_th_values.size()) *
                                  static_cast<int>(config.evaluation.alpha_values.size());
        const int total_outer_runs = static_cast<int>(config.evaluation.datasets.size()) *
                                     static_cast<int>(config.evaluation.encodings.size()) *
                                     config.experiment.repeats * (1 + snn_per_combo);

        // Overall-progress banner across the whole 4-profile run. Each profile is a separate
        // process, so this process cannot know the outer progress on its own — the wrapper
        // (01_e04_run_article_profiles.sh) computes it the same way run_e05_profiles.sh does
        // (work-weighted, EMA-smoothed seconds-per-unit-work — see scripts/lib/run_eta.sh) and
        // passes the ready-made line in via E04_OVERALL. Logging it renders it as a persistent
        // top line above the per-profile bars; empty/unset when run standalone, so unchanged.
        if (const char* overall = std::getenv("E04_OVERALL");
            overall != nullptr && overall[0] != '\0')
        {
            nn::progress::ProgressManager::instance().log(std::string(overall));
        }

        const uint32_t run_bar = nn::progress::ProgressManager::instance().create_bar(
            "Profile: " + config.experiment.run_tag, static_cast<float>(total_outer_runs));
        nn::progress::ProgressManager::instance().set_description(
            run_bar, "SNN vs LSTM comparative experiment");

        int completed_runs = 0;

        for (const auto& dataset_name : config.evaluation.datasets)
        {
            const DatasetSplit split = build_split(config, dataset_name);

            for (const auto& encoding : config.evaluation.encodings)
            {
                for (int run_id = 0; run_id < config.experiment.repeats; ++run_id)
                {
                    (void) completed_runs; // updated inside each LSTM/SNN block
                    const std::uint32_t run_seed =
                        config.experiment.seed_deterministic
                            ? config.experiment.seed
                            : config.experiment.seed + static_cast<std::uint32_t>(run_id);

                    {
                        const CheckpointKey lstm_key{config.experiment.run_tag,
                            backend_name,
                            dataset_name,
                            "lstm-ae",
                            encoding,
                            "lstm",
                            0.0f,
                            0.0f,
                            run_id + 1};
                        const auto lstm_chk = checkpoint_path(chk_dir, lstm_key);

                        if (checkpoint_is_valid(lstm_chk, cfg_hash))
                        {
                            all_rows.push_back(checkpoint_load(lstm_chk));
                            nn::progress::ProgressManager::instance().update_bar(
                                run_bar, static_cast<float>(++completed_runs));
                        }
                        else
                        {
                            float train_ms = 0.0f;
                            float infer_ms = 0.0f;
                            auto lstm_cfg = make_lstm_cfg(config);

                            nn::models::lstm::LSTMAutoencoder lstm_model(lstm_cfg);

                            TrainResult train_result = train_with_early_stopping_lstm( //
                                lstm_model,                                            //
                                config,                                                //
                                split.train_samples,                                   //
                                split.val_samples,                                     //
                                encoding,                                              //
                                run_seed,                                              //
                                static_cast<std::size_t>(run_id),                      //
                                static_cast<std::size_t>(config.experiment.repeats),   //
                                train_ms,                                              //
                                infer_ms                                               //
                            );
                            RunMetrics metrics = train_result.metrics;
                            metrics.train_ms = train_ms;

                            if (!config.dataset.latex_data_dir.empty())
                            {
                                const std::filesystem::path latex_dir =
                                    std::filesystem::path(config.dataset.latex_data_dir);
                                write_epoch_history_dat(
                                    latex_dir /
                                        (config.experiment.run_tag + "_lstm_" + encoding + "_run" +
                                            std::to_string(run_id + 1) + "_history.dat"),
                                    "lstm-ae",
                                    encoding,
                                    "",
                                    0.0f,
                                    0.0f,
                                    run_id + 1,
                                    train_result.history);
                                write_batch_convergence_dat(
                                    latex_dir /
                                        (config.experiment.run_tag + "_lstm_" + encoding + "_run" +
                                            std::to_string(run_id + 1) + "_convergence.dat"),
                                    "lstm-ae",
                                    encoding,
                                    "",
                                    0.0f,
                                    0.0f,
                                    run_id + 1,
                                    train_result.history);
                            }

                            if (config.dataset.save_models)
                            {
                                const std::string base_name = sanitize_name(
                                    config.experiment.run_tag + "_lstm_" + dataset_name + "_" +
                                    encoding + "_run" + std::to_string(run_id + 1));
                                const std::filesystem::path state_txt =
                                    models_dir / (base_name + "_state_dict.txt");
                                const bool ok =
                                    save_state_dict_text(state_txt, lstm_model.state_dict());
                                if (!ok)
                                {
                                    NN_LOG_WARN(
                                        "[comparative] failed to save LSTM state_dict for " +
                                        base_name);
                                }
                            }

                            all_rows.push_back( //
                                ResultRow{
                                    backend_name,              //
                                    config.experiment.run_tag, //
                                    dataset_name,              //
                                    "lstm-ae",                 //
                                    encoding,                  //
                                    "lstm",                    //
                                    1,                         //
                                    0.0f,                      //
                                    0.0f,                      //
                                    run_id + 1,                //
                                    run_seed,                  //
                                    cfg_hash,                  //
                                    metrics                    //
                                } //
                            );
                            checkpoint_save(
                                lstm_chk, all_rows.back(), train_result.history, cfg_hash);
                            nn::progress::ProgressManager::instance().update_bar(
                                run_bar, static_cast<float>(++completed_runs));
                        }
                    }

                    for (const auto& architecture : config.evaluation.snn_architectures)
                    {
                        for (float voltage_threshold : config.evaluation.v_th_values)
                        {
                            for (float alpha : config.evaluation.alpha_values)
                            {
                                const CheckpointKey snn_key{config.experiment.run_tag,
                                    backend_name,
                                    dataset_name,
                                    "snn-ae",
                                    encoding,
                                    architecture,
                                    voltage_threshold,
                                    alpha,
                                    run_id + 1};
                                const auto snn_chk = checkpoint_path(chk_dir, snn_key);

                                if (checkpoint_is_valid(snn_chk, cfg_hash))
                                {
                                    all_rows.push_back(checkpoint_load(snn_chk));
                                    nn::progress::ProgressManager::instance().update_bar(
                                        run_bar, static_cast<float>(++completed_runs));
                                }
                                else
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
                                            extract_layer_sizes(config.model.encoder_layer_spec)
                                                    .empty()
                                                ? 0
                                                : extract_layer_sizes(
                                                      config.model.encoder_layer_spec)
                                                      .front()) +
                                        "|" + std::to_string(voltage_threshold) + "|" +
                                        std::to_string(alpha);

                                    ProtocolSpikingAutoencoder snn_model(snn_config);

                                    TrainResult train_result = train_with_early_stopping_snn( //
                                        snn_model,                                            //
                                        config,                                               //
                                        split.train_samples,                                  //
                                        split.val_samples,                                    //
                                        split.val_labels,                                     //
                                        encoding,                                             //
                                        architecture,                                         //
                                        alpha,                                                //
                                        voltage_threshold,                                    //
                                        run_seed,                                             //
                                        static_cast<std::size_t>(run_id),                     //
                                        static_cast<std::size_t>(config.experiment.repeats),  //
                                        train_ms,                                             //
                                        infer_ms                                              //
                                    );

                                    RunMetrics metrics = train_result.metrics;
                                    metrics.train_ms = train_ms;

                                    if (!config.dataset.latex_data_dir.empty())
                                    {
                                        const std::filesystem::path latex_dir =
                                            std::filesystem::path(config.dataset.latex_data_dir);
                                        write_epoch_history_dat(
                                            latex_dir /
                                                (config.experiment.run_tag + "_snn_" + encoding +
                                                    "_" + architecture + "_vth" +
                                                    std::to_string(voltage_threshold) + "_a" +
                                                    std::to_string(alpha) + "_run" +
                                                    std::to_string(run_id + 1) + "_history.dat"),
                                            "snn-ae",
                                            encoding,
                                            architecture,
                                            voltage_threshold,
                                            alpha,
                                            run_id + 1,
                                            train_result.history);
                                        write_batch_convergence_dat(
                                            latex_dir / (config.experiment.run_tag + "_snn_" +
                                                            encoding + "_" + architecture + "_vth" +
                                                            std::to_string(voltage_threshold) +
                                                            "_a" + std::to_string(alpha) + "_run" +
                                                            std::to_string(run_id + 1) +
                                                            "_convergence.dat"),
                                            "snn-ae",
                                            encoding,
                                            architecture,
                                            voltage_threshold,
                                            alpha,
                                            run_id + 1,
                                            train_result.history);
                                    }

                                    if (config.dataset.save_models)
                                    {
                                        const std::string base_name = sanitize_name(
                                            config.experiment.run_tag + "_snn_" + dataset_name +
                                            "_" + encoding + "_" + architecture + "_vth" +
                                            std::to_string(voltage_threshold) + "_a" +
                                            std::to_string(alpha) + "_run" +
                                            std::to_string(run_id + 1));
                                        const std::filesystem::path encoder_txt =
                                            models_dir / (base_name + "_encoder_params.txt");
                                        const std::filesystem::path decoder_txt =
                                            models_dir / (base_name + "_decoder_params.txt");
                                        const bool enc_ok = save_parameter_list_text(
                                            encoder_txt, snn_model.encoder_.params(), "encoder");
                                        const bool dec_ok = save_parameter_list_text(
                                            decoder_txt, snn_model.decoder_.params(), "decoder");
                                        if (!enc_ok || !dec_ok)
                                        {
                                            NN_LOG_WARN(
                                                "[comparative] failed to save SNN model artifacts "
                                                "for " +
                                                base_name);
                                        }
                                    }

                                    all_rows.push_back( //
                                        ResultRow{
                                            backend_name,              //
                                            config.experiment.run_tag, //
                                            dataset_name,              //
                                            "snn-ae",                  //
                                            encoding,                  //
                                            architecture,              //
                                            static_cast<int>(
                                                config.model.encoder_layer_spec.size()), //
                                            voltage_threshold,                           //
                                            alpha,                                       //
                                            run_id + 1,                                  //
                                            run_seed,                                    //
                                            cfg_hash,                                    //
                                            metrics                                      //
                                        } //
                                    );
                                    checkpoint_save(
                                        snn_chk, all_rows.back(), train_result.history, cfg_hash);
                                    nn::progress::ProgressManager::instance().update_bar(
                                        run_bar, static_cast<float>(++completed_runs));
                                }
                            }
                        }
                    }
                }
            }
        }

        nn::progress::ProgressManager::instance().complete_bar(run_bar);
        nn::progress::ProgressManager::instance().shutdown();

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

        if (!config.dataset.latex_data_dir.empty())
        {
            const std::filesystem::path latex_dir =
                std::filesystem::path(config.dataset.latex_data_dir);
            write_latex_exports(latex_dir, config.experiment.run_tag, config, all_rows);
            write_pgfplots_summary_dat(
                latex_dir / (config.experiment.run_tag + "_summary.dat"), all_rows);
            write_pgfplots_sweep_dat(
                latex_dir / (config.experiment.run_tag + "_sweep.dat"), all_rows);
        }

        NN_LOG_INFO("[comparative] Results written to: " + csv_path.string() + ", " +
                    table_path.string() + ", " + summary_json.string());

        flushProgressAsync();
        return 0;
    }
    catch (const std::exception& ex)
    {
        NN_LOG_ERROR(std::string("[comparative] Fatal error: ") + ex.what());
        return 1;
    }
}

} // namespace e04
