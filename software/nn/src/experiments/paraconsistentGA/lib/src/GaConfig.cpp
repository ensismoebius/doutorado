#include "GaConfig.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "StrictJson.hpp"

namespace pga
{

double GaConfig::autoencoder_budget_ms() const
{
    return std::max(0.0, constraints.latency_ceiling_ms - constraints.fixed_pipeline_cost_ms);
}

namespace
{

using C = const GaConfig&;

constexpr const char* kAlways = "always";
auto always(C /*unused*/) -> bool
{
    return true;
}
auto snn(C c) -> bool
{
    return c.is_snn_population();
}

// The GA's own keys, one row each with the condition under which the GA reads it (the
// same table form as ThesisConfig.cpp's for_each_field).
template <typename Cfg, typename Row>
void for_each_field(Cfg& c, Row&& row)
{
    row(c.run_tag, kAlways, always);
    row(c.results_dir, kAlways, always);

    auto& g = c.ga;
    row(g.population_size, kAlways, always);
    row(g.generations, kAlways, always);
    row(g.n_seeds, kAlways, always);
    row(g.n_losers, kAlways, always);
    row(g.crossover_prob, kAlways, always);
    row(g.mutation_prob, kAlways, always);
    row(g.seed, kAlways, always);
    row(g.tournament_k, kAlways, always);

    auto& b = g.bounds;
    row(b.min_layers, kAlways, always);
    row(b.max_layers, kAlways, always);
    row(b.min_width, kAlways, always);
    row(b.max_width, kAlways, always);
    constexpr const char* kSnn = "the autoencoder model is \"snn-ae\"";
    row(b.encoding_choices, kSnn, snn);
    row(b.evolve_temporal, kSnn, snn);
    constexpr const char* kTemporal =
        "the model is \"snn-ae\" and ga.bounds.evolve_temporal is true";
    const auto temporal = [](C k) { return snn(k) && k.ga.bounds.evolve_temporal; };
    row(b.time_steps_choices, kTemporal, temporal);
    row(b.voltage_threshold_min, kTemporal, temporal);
    row(b.voltage_threshold_max, kTemporal, temporal);

    auto& k = c.constraints;
    row(k.latency_ceiling_ms, kAlways, always);
    row(k.fixed_pipeline_cost_ms, kAlways, always);
    row(k.latency_calibrated, kAlways, always);
    row(k.ns_per_mac, kAlways, always);
    row(k.tau_rec, kAlways, always);

    row(c.checkpoint.enabled, kAlways, always);
    row(c.checkpoint.every_generations,
        "checkpoint.enabled is true",
        [](C x) { return x.checkpoint.enabled.get(); });
}

const std::string kOwner = "GaConfig";

// The thesis config as one genome would run it: the genome-owned autoencoder keys filled in
// by the same to_ae_config the fitness uses.
auto genome_config(const GaConfig& cfg, const Genome& g) -> thesis::ThesisConfig
{
    thesis::ThesisConfig run = cfg.base;
    run.feature_extraction.autoencoder = to_ae_config(g, cfg.base.feature_extraction.autoencoder);
    return run;
}

} // namespace

auto GaConfig::embedding() -> thesis::ThesisConfig::Embedding
{
    using Key = thesis::ThesisConfig::Embedding::Key;
    thesis::ThesisConfig::Embedding e;
    e.program = "paraconsistentGA";
    e.own_sections = {"run_tag", "results_dir", "ga", "constraints", "checkpoint"};
    e.set_by_program = {
        Key{"feature_extraction.autoencoder.encoder_layer_spec",
            "sets it from each genome (the evolved widths)"},
        Key{"feature_extraction.autoencoder.decoder_layer_spec",
            "sets it from each genome (the encoder's mirror)"},
        Key{"feature_extraction.autoencoder.encoding",
            "sets it from each genome (drawn from ga.bounds.encoding_choices)"},
        Key{"feature_extraction.autoencoder.time_steps",
            "sets it from each genome (evolved, or coupled to the encoding)"},
        Key{"feature_extraction.autoencoder.voltage_threshold",
            "sets it from each genome (evolved, or coupled to the encoding)"},
    };
    e.not_read = {
        Key{"experiment.run_tag", "names its outputs with the top-level run_tag"},
        Key{"experiment.repeats", "repeats each genome through ga.n_seeds instead"},
        Key{"experiment.seed_deterministic", "repeats each genome through ga.n_seeds instead"},
        Key{"dataset.results_dir", "writes to the top-level results_dir"},
    };
    return e;
}

namespace
{
// The GA evolves autoencoders: every GA rule that depends on the model needs this first.
void require_autoencoder(const thesis::ThesisConfig& base)
{
    if (base.feature_extraction.strategy != "autoencoder")
        throw std::invalid_argument(
            "GaConfig: feature_extraction.strategy must be \"autoencoder\" — the GA evolves "
            "autoencoder architectures; got \"" +
            base.feature_extraction.strategy.get() + "\"");
}
} // namespace

void GaConfig::validate() const
{
    require_autoencoder(base);
    for_each_field(*this,
        [&](const auto& field, const char* when, auto applies)
        { thesis::strict::check_declared(field, applies(*this), when, kOwner); });

    // The GA ranks autoencoder features before any classifier exists (Phase-00-like). The
    // profile states this rather than having it overwritten: an override would make a
    // profile that says otherwise run something it does not say.
    if (base.classifier.enabled)
        throw std::invalid_argument(
            "GaConfig: classifier.enabled must be false — the GA scores features by "
            "paraconsistent d_penalized and never trains the authentication head");
    if (!base.paraconsistent.enabled)
        throw std::invalid_argument(
            "GaConfig: paraconsistent.enabled must be true — d_penalized is the GA's objective");
    if (base.paraconsistent.speaker_subsets)
        throw std::invalid_argument(
            "GaConfig: paraconsistent.speaker_subsets must be false — the GA scores each "
            "candidate on the whole set and never writes the per-subset table; score the "
            "winner's subsets with a thesis Phase 00 profile instead");

    const auto& model = base.feature_extraction.autoencoder.model;
    if (model != "snn-ae" && model != "ann-ae")
        throw std::invalid_argument(
            "GaConfig: autoencoder.model must be snn-ae or ann-ae (lstm-ae not supported by the "
            "GA)");

    if (ga.population_size < 4)
        throw std::invalid_argument("GaConfig: ga.population_size must be >= 4");
    if (ga.generations < 1) throw std::invalid_argument("GaConfig: ga.generations must be >= 1");
    if (ga.n_seeds < 1) throw std::invalid_argument("GaConfig: ga.n_seeds must be >= 1");
    if (ga.n_losers < 0) throw std::invalid_argument("GaConfig: ga.n_losers must be >= 0");
    if (ga.n_losers >= ga.population_size)
        throw std::invalid_argument(
            "GaConfig: ga.n_losers (" + std::to_string(ga.n_losers.get()) +
            ") must be < population_size (" + std::to_string(ga.population_size.get()) +
            "); at least one elite slot must remain, otherwise selection keeps no winners.");
    if (ga.seed == 0u) throw std::invalid_argument("GaConfig: ga.seed must be non-zero");
    if (ga.crossover_prob < 0.0 || ga.crossover_prob > 1.0)
        throw std::invalid_argument("GaConfig: ga.crossover_prob must be in [0,1]");
    if (ga.mutation_prob < 0.0 || ga.mutation_prob > 1.0)
        throw std::invalid_argument("GaConfig: ga.mutation_prob must be in [0,1]");
    if (ga.tournament_k < 2) throw std::invalid_argument("GaConfig: ga.tournament_k must be >= 2");

    // Free-architecture bounds: depth range and per-layer width range.
    if (ga.bounds.min_width < 1)
        throw std::invalid_argument("GaConfig: ga.bounds.min_width must be >= 1");
    if (ga.bounds.max_width < ga.bounds.min_width)
        throw std::invalid_argument("GaConfig: ga.bounds.max_width must be >= min_width");
    if (ga.bounds.min_layers < 1)
        throw std::invalid_argument("GaConfig: ga.bounds.min_layers must be >= 1");
    if (ga.bounds.max_layers < ga.bounds.min_layers)
        throw std::invalid_argument("GaConfig: ga.bounds.max_layers must be >= min_layers");
    // Strictly decreasing widths need one distinct integer per layer, so the deepest
    // requested genome must fit in the width range. Reject rather than silently
    // truncate every deep genome to a shallower one.
    const int width_room = ga.bounds.max_width - ga.bounds.min_width + 1;
    if (ga.bounds.max_layers > width_room)
        throw std::invalid_argument(
            "GaConfig: ga.bounds.max_layers (" + std::to_string(ga.bounds.max_layers.get()) +
            ") exceeds the number of distinct widths available (" + std::to_string(width_room) +
            " = max_width - min_width + 1). Strictly decreasing layers cannot be formed; "
            "widen the width range or reduce max_layers.");
    if (is_snn_population())
    {
        if (ga.bounds.encoding_choices->empty())
            throw std::invalid_argument("GaConfig: ga.bounds.encoding_choices must be non-empty");
        if (ga.bounds.evolve_temporal)
        {
            if (ga.bounds.time_steps_choices->empty())
                throw std::invalid_argument(
                    "GaConfig: ga.bounds.time_steps_choices must be non-empty");
            if (!(ga.bounds.voltage_threshold_min > 0.0f) ||
                ga.bounds.voltage_threshold_max < ga.bounds.voltage_threshold_min)
                throw std::invalid_argument(
                    "GaConfig: require 0 < ga.bounds.voltage_threshold_min <= "
                    "voltage_threshold_max");
        }
    }

    if (constraints.latency_ceiling_ms <= 0.0)
        throw std::invalid_argument("GaConfig: constraints.latency_ceiling_ms must be > 0");
    if (constraints.fixed_pipeline_cost_ms < 0.0)
        throw std::invalid_argument("GaConfig: constraints.fixed_pipeline_cost_ms must be >= 0");
    if (constraints.ns_per_mac <= 0.0)
        throw std::invalid_argument("GaConfig: constraints.ns_per_mac must be > 0");
    if (constraints.tau_rec < 0.0)
        throw std::invalid_argument("GaConfig: constraints.tau_rec must be >= 0");
    if (autoencoder_budget_ms() <= 0.0)
        throw std::invalid_argument(
            "GaConfig: fixed_pipeline_cost_ms >= latency_ceiling_ms leaves the autoencoder no "
            "budget");

    if (checkpoint.enabled && checkpoint.every_generations < 1)
        throw std::invalid_argument("GaConfig: checkpoint.every_generations must be >= 1");

    if (run_tag->empty()) throw std::invalid_argument("GaConfig: run_tag must not be empty");
    if (results_dir->empty())
        throw std::invalid_argument("GaConfig: results_dir must not be empty");

    // The embedded thesis config, validated as the GA will run it: with a genome's keys
    // filled in, once per encoding the search may draw (and, when T and V_th are genes,
    // per T choice at both threshold bounds). This is where an encoding choice that breaks
    // the encoding<->loss invariant is caught — e.g. ae_loss_type spikecount with "direct"
    // among the choices — instead of failing (or not) mid-search.
    Genome probe;
    probe.encoder_widths = {ga.bounds.max_width};
    if (!is_snn_population())
    {
        probe.encoding = "direct";
        probe.time_steps = 1;
        probe.voltage_threshold = 1.0f;
        genome_config(*this, probe).validate();
        return;
    }
    for (const auto& encoding : ga.bounds.encoding_choices)
    {
        probe.encoding = encoding;
        if (!ga.bounds.evolve_temporal)
        {
            apply_phase00_temporal_coupling(probe);
            genome_config(*this, probe).validate();
            continue;
        }
        for (const int t : ga.bounds.time_steps_choices)
            for (const float vth :
                {ga.bounds.voltage_threshold_min.get(), ga.bounds.voltage_threshold_max.get()})
            {
                probe.time_steps = t;
                probe.voltage_threshold = vth;
                genome_config(*this, probe).validate();
            }
    }
}

GaConfig GaConfig::from_json(const nlohmann::json& j)
{
    GaConfig cfg;
    cfg.base = thesis::ThesisConfig::from_json(j, embedding());
    require_autoencoder(cfg.base);

    std::set<std::string> fields;
    for_each_field(cfg,
        [&](auto& field, const char* /*when*/, auto /*applies*/) { fields.insert(field.path()); });
    // Visit only the GA's own keys; the thesis sections were checked by the thesis parser.
    const std::vector<std::string> thesis_sections = {
        "experiment", "dataset", "feature_extraction", "paraconsistent", "classifier", "training"};
    thesis::strict::refuse_unknown_keys(
        j, "", fields, thesis::strict::sections_of(fields), thesis_sections, kOwner);

    for_each_field(cfg,
        [&](auto& field, const char* /*when*/, auto /*applies*/)
        {
            using T = typename std::remove_cvref_t<decltype(field)>::value_type;
            if (const auto* v = thesis::strict::find_path(j, field.path()))
                field = thesis::strict::read_exact<T>(*v, kOwner, field.path());
        });

    for_each_field(cfg,
        [&](const auto& field, const char* when, auto applies)
        { thesis::strict::check_declared(field, applies(cfg), when, kOwner); });
    return cfg;
}

GaConfig GaConfig::from_file(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("GaConfig: cannot open " + path);
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

} // namespace pga
