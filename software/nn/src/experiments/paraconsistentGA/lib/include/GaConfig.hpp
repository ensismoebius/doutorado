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
// training, base autoencoder model + modality — reused verbatim, see
// .wiki/Experiments/ParaconsistentGA-Design.md §2) and adds the GA-specific "ga" block. One run
// evolves ONE population, defined by base.feature_extraction.autoencoder.model ∈ {snn-ae, ann-ae}
// and base.dataset.modality ∈ {eeg, voice, fused} (.wiki/Experiments/ParaconsistentGA-Design.md
// §5.3/§5.4).
//
// NO FIELD HAS A DEFAULT (thesis::Required, as in ThesisConfig): from_json refuses a profile
// that omits a key the run reads, declares one it ignores, or contains an unknown key.
struct GaConfig
{
    thesis::ThesisConfig base;

    struct Ga
    {
        thesis::Required<int> population_size{"ga.population_size"}; ///< >= 4
        thesis::Required<int> generations{"ga.generations"};         ///< >= 1

        // Each individual is trained under n_seeds distinct seeds; its d_penalized is
        // the mean, and the std is recorded as the stability measure
        // (.wiki/Experiments/ParaconsistentGA-Design.md §5.5).
        thesis::Required<int> n_seeds{"ga.n_seeds"};

        // Survivor selection reserves this many slots for deliberately-kept LOSERS —
        // the worst-ranked, most distinct non-survivors — instead of filling the whole
        // next generation with the μ+λ elite. Losing genetic material is a hedge against
        // premature convergence to a local optimum: it can recombine into something good
        // generations later. 0 recovers textbook NSGA-II. Must be < population_size so
        // at least one elite slot remains.
        thesis::Required<int> n_losers{"ga.n_losers"};

        // Meiotic recombination probability. Reused as the per-meiosis chance that a
        // gamete is a crossover of the parent's two haplotypes rather than a straight
        // copy of one (see GaGenome meiosis()).
        thesis::Required<double> crossover_prob{"ga.crossover_prob"};
        thesis::Required<double> mutation_prob{"ga.mutation_prob"};
        thesis::Required<std::uint32_t> seed{
            "ga.seed"}; ///< GA-level RNG (init/selection/variation)

        thesis::Required<int> tournament_k{"ga.tournament_k"}; ///< 2 = binary tournament

        GenomeBounds bounds;
    } ga;

    struct Constraints
    {
        // End-to-end ceiling on the target hardware (.wiki/Experiments/ParaconsistentGA-Design.md
        // §4). NEW requirement — not a previously established project constraint.
        thesis::Required<double> latency_ceiling_ms{"constraints.latency_ceiling_ms"};

        // Constant, genome-independent cost of the rest of the chain (acquisition,
        // preprocessing, feature assembly, response). autoencoder_budget =
        // latency_ceiling_ms - fixed_pipeline_cost_ms.
        thesis::Required<double> fixed_pipeline_cost_ms{"constraints.fixed_pipeline_cost_ms"};

        // False → fixed_pipeline_cost_ms and ns_per_mac are uncalibrated estimates;
        // logged as such (.wiki/Experiments/ParaconsistentGA-Design.md §4).
        thesis::Required<bool> latency_calibrated{"constraints.latency_calibrated"};

        // MAC→time conversion for the structural latency proxy.
        // est_autoencoder_ms = inference_cost_proxy · ns_per_mac / 1e6.
        thesis::Required<double> ns_per_mac{"constraints.ns_per_mac"};

        // §3.3 sanity filter, implemented as a latent-collapse guard (see PHASE0.md):
        // an individual is infeasible if the mean per-dimension std of its latent
        // vectors is below this. Defends against the α=β=1 constant-latent degeneracy.
        // Separate value per population; never compared across ANN/SNN
        // (.wiki/Experiments/ParaconsistentGA-Design.md §5.3).
        thesis::Required<double> tau_rec{"constraints.tau_rec"};
    } constraints;

    // Crash-resilient checkpointing (results/paraconsistentGA/pga_<tag>_{checkpoint.json,
    // cache.jsonl}). Two layers: the per-individual cache (every trained genome appended
    // the instant it is scored, so no genome is ever retrained across restarts) and the
    // per-generation state (population + RNG + generation index, so the loop resumes
    // exactly). On a crash the loss is at most one in-flight training. Both artifacts are
    // deleted on successful completion — the results (CSV + Pareto JSON) are what remain.
    struct Checkpoint
    {
        thesis::Required<bool> enabled{"checkpoint.enabled"};
        // enabled only: write the generation-state file every N generations. The
        // per-individual cache is always appended live regardless of this.
        thesis::Required<int> every_generations{"checkpoint.every_generations"};
    } checkpoint;

    // Where per-individual logs and the Pareto front are written, and their name. The
    // GA's own keys: the embedded experiment.run_tag / dataset.results_dir are refused.
    thesis::Required<std::string> results_dir{"results_dir"};
    thesis::Required<std::string> run_tag{"run_tag"};

    [[nodiscard]] bool is_snn_population() const
    {
        return base.feature_extraction.autoencoder.model == "snn-ae";
    }

    // Budget the autoencoder actually has (ms). Clamped at 0.
    [[nodiscard]] double autoencoder_budget_ms() const;

    /// The thesis keys this program owns or ignores (see thesis::ThesisConfig::Embedding).
    static auto embedding() -> thesis::ThesisConfig::Embedding;

    void validate() const;

    static GaConfig from_json(const nlohmann::json& j);
    static GaConfig from_file(const std::string& path);
};

} // namespace pga
