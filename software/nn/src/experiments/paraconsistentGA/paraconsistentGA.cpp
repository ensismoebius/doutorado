/**
 * @file src/experiments/paraconsistentGA/paraconsistentGA.cpp
 * @brief NSGA-II architecture search over autoencoder feature extractors, ranked by
 *        paraconsistent feature quality (d_penalized) under a latency constraint.
 *
 * One run evolves ONE population, defined by the profile's autoencoder.model
 * (snn-ae | ann-ae) and dataset.modality (eeg | voice | fused). See ga.md / PHASE0.md.
 *
 * Usage:
 *   paraconsistentGA --config <profile.json>
 */

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "ThesisDataset.hpp"
#include "lib/include/GaConfig.hpp"
#include "lib/include/GaNsga2.hpp"
#include "lib/include/GaOutput.hpp"
#include "progress/ProgressManager.hpp"
#include "utility/progress.hpp"

namespace
{
void print_usage(const char* prog)
{
    std::cerr << "Usage: " << prog << " --config <profile.json>\n";
}

std::string parse_config_path(int argc, char* argv[])
{
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--config") return std::string(argv[i + 1]);
    return {};
}
} // namespace

auto main(int argc, char* argv[]) -> int
{
    try
    {
        const std::string config_path = parse_config_path(argc, argv);
        if (config_path.empty())
        {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        auto cfg = pga::GaConfig::from_file(config_path);
        cfg.validate();

        auto& pm = nn::progress::ProgressManager::instance();

        // Cross-profile progress banner. Each profile is a separate process and cannot
        // know the whole-sweep progress on its own, so the runner
        // (01_paraconsistentGA_run_all_profiles.sh) computes it and passes the finished
        // line in via PGA_OVERALL. Logged first, it renders as a persistent top line
        // above the per-generation bars (the Guayaquil/Thesis convention). Empty/unset
        // when run standalone, so the single-profile TUI is unchanged.
        if (const char* overall = std::getenv("PGA_OVERALL");
            overall != nullptr && overall[0] != '\0')
        {
            pm.log(std::string(overall));
        }

        pm.log("[PGA] run_tag=" + cfg.run_tag +
               " model=" + cfg.base.feature_extraction.autoencoder.model + " modality=" +
               cfg.base.dataset.modality + " pop=" + std::to_string(cfg.ga.population_size) +
               " gens=" + std::to_string(cfg.ga.generations) +
               " seeds=" + std::to_string(cfg.ga.n_seeds));
        if (!cfg.constraints.latency_calibrated)
            pm.log(
                "[PGA] WARNING: latency proxy is UNCALIBRATED — fixed_pipeline_cost_ms and "
                "ns_per_mac are estimates, not measured on target hardware (ga.md §4).");

        // Dataset loaded once; seed-independent (reused thesis loader).
        auto view = thesis::load_dataset(cfg.base.dataset);
        pm.log("[PGA] Loaded " + std::to_string(view.samples.size()) + " samples from " +
               std::to_string(view.n_subjects) + " subjects.");

        const auto result = pga::run_nsga2(view, cfg, nullptr);

        pga::write_individuals_csv(cfg.results_dir, cfg.run_tag, result.history);
        pga::write_pareto_front_json(cfg.results_dir, cfg.run_tag, cfg, result);

        pm.log("[PGA] Done. Evaluated " + std::to_string(result.history.size()) +
               " distinct genomes; Pareto front has " + std::to_string(result.pareto_front.size()) +
               " feasible members. Results in " + cfg.results_dir);

        pm.shutdown();
        flushProgressAsync();
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[PGA] Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
