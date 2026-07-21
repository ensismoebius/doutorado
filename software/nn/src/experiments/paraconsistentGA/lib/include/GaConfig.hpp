#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "GaGenome.hpp"
#include "ThesisConfig.hpp"
#include "nlohmann/json.hpp"

namespace pga
{

// Configuration for one GA run. Embeds a full thesis::ThesisConfig (dataset,
// training, base autoencoder model + modality — reused verbatim, see ga.md §2) and
// adds the GA-specific "ga" block. One run evolves ONE population, defined by
// base.feature_extraction.autoencoder.model ∈ {snn-ae, ann-ae} and
// base.dataset.modality ∈ {eeg, voice, fused} (ga.md §5.3/§5.4).
struct GaConfig
{
    thesis::ThesisConfig base;

    struct Ga
    {
        int population_size = 16;
        int generations = 12;

        // Each individual is trained under n_seeds distinct seeds; its d_penalized is
        // the mean, and the std is recorded as the stability measure (ga.md §5.5).
        int n_seeds = 3;

        double crossover_prob = 0.9;
        double mutation_prob = 0.2;
        std::uint32_t seed = 42u; // GA-level RNG seed (init/selection/variation).

        int tournament_k = 2; // binary tournament (standard NSGA-II).

        GenomeBounds bounds;
    } ga;

    struct Constraints
    {
        // End-to-end ceiling on the target hardware (ga.md §4). NEW requirement —
        // not a previously established project constraint.
        double latency_ceiling_ms = 1000.0;

        // Constant, genome-independent cost of the rest of the chain (acquisition,
        // preprocessing, feature assembly, response). autoencoder_budget =
        // latency_ceiling_ms - fixed_pipeline_cost_ms.
        double fixed_pipeline_cost_ms = 0.0;

        // False → fixed_pipeline_cost_ms and ns_per_mac are uncalibrated estimates;
        // logged as such (ga.md §4). A real calibration utility would set true.
        bool latency_calibrated = false;

        // Uncalibrated MAC→time conversion for the structural latency proxy.
        // est_autoencoder_ms = inference_cost_proxy · ns_per_mac / 1e6.
        double ns_per_mac = 2.0;

        // §3.3 sanity filter, implemented as a latent-collapse guard (see PHASE0.md):
        // an individual is infeasible if the mean per-dimension std of its latent
        // vectors is below this. Defends against the α=β=1 constant-latent degeneracy.
        // Separate value per population; never compared across ANN/SNN (ga.md §5.3).
        double tau_rec = 1e-4;
    } constraints;

    // Where per-individual logs and the Pareto front are written.
    std::string results_dir = "results/paraconsistentGA";
    std::string run_tag = "pga_run";

    [[nodiscard]] bool is_snn_population() const
    {
        return base.feature_extraction.autoencoder.model == "snn-ae";
    }

    // Budget the autoencoder actually has (ms). Clamped at 0.
    [[nodiscard]] double autoencoder_budget_ms() const;

    void validate() const;

    static GaConfig from_json(const nlohmann::json& j);
    static GaConfig from_file(const std::string& path);
};

} // namespace pga
