#include "GaGenome.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pga
{

namespace
{
// Pick a uniformly random element from a non-empty choice list.
template <typename T>
T pick(std::mt19937& rng, const std::vector<T>& choices)
{
    std::uniform_int_distribution<size_t> d(0, choices.size() - 1);
    return choices[d(rng)];
}

// Enforce a real bottleneck: latent strictly below hidden. If violated, drop latent to
// the largest legal choice below hidden. Raises when the bounds admit none — a silently
// invented width would hide a misconfigured search space.
void repair_bottleneck(Genome& g, const GenomeBounds& bounds)
{
    if (g.latent < g.hidden) return;
    int best = 0;
    for (int c : bounds.latent_choices)
        if (c < g.hidden && c > best) best = c;
    if (best <= 0)
        throw std::invalid_argument(
            "GaGenome: no latent_choices value is smaller than hidden=" + std::to_string(g.hidden) +
            " — cannot form a bottleneck. Fix ga.bounds rather than silently halving "
            "the width.");
    g.latent = best;
}
} // namespace

void apply_phase00_temporal_coupling(Genome& g)
{
    if (g.encoding == "direct")
    {
        g.time_steps = 1;
        g.voltage_threshold = 1.0f;
    }
    else // latency | poisson
    {
        g.time_steps = 16;
        g.voltage_threshold = 0.2f;
    }
}

Genome random_genome(std::mt19937& rng, const GenomeBounds& bounds, bool is_snn)
{
    Genome g;
    g.hidden = pick(rng, bounds.hidden_choices);
    g.latent = pick(rng, bounds.latent_choices);
    repair_bottleneck(g, bounds);

    if (is_snn)
    {
        g.encoding = pick(rng, bounds.encoding_choices);
        if (bounds.evolve_temporal)
        {
            g.time_steps = pick(rng, bounds.time_steps_choices);
            std::uniform_real_distribution<float> vd(
                bounds.voltage_threshold_min, bounds.voltage_threshold_max);
            g.voltage_threshold = vd(rng);
        }
        else
        {
            apply_phase00_temporal_coupling(g);
        }
    }
    else
    {
        g.encoding = "direct";
        g.time_steps = 1;
        g.voltage_threshold = 1.0f;
    }
    return g;
}

Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng, bool is_snn)
{
    std::bernoulli_distribution coin(0.5);
    Genome c;
    c.hidden = coin(rng) ? a.hidden : b.hidden;
    c.latent = coin(rng) ? a.latent : b.latent;
    if (is_snn)
    {
        c.encoding = coin(rng) ? a.encoding : b.encoding;
        c.time_steps = coin(rng) ? a.time_steps : b.time_steps;
        c.voltage_threshold = coin(rng) ? a.voltage_threshold : b.voltage_threshold;
    }
    else
    {
        c.encoding = "direct";
        c.time_steps = 1;
        c.voltage_threshold = 1.0f;
    }
    return c;
}

void mutate(Genome& g, std::mt19937& rng, const GenomeBounds& bounds, double prob, bool is_snn)
{
    std::bernoulli_distribution hit(prob);
    if (hit(rng)) g.hidden = pick(rng, bounds.hidden_choices);
    if (hit(rng)) g.latent = pick(rng, bounds.latent_choices);
    repair_bottleneck(g, bounds);

    if (!is_snn)
    {
        g.encoding = "direct";
        g.time_steps = 1;
        g.voltage_threshold = 1.0f;
        return;
    }

    if (hit(rng)) g.encoding = pick(rng, bounds.encoding_choices);
    if (bounds.evolve_temporal)
    {
        if (hit(rng)) g.time_steps = pick(rng, bounds.time_steps_choices);
        if (hit(rng))
        {
            std::uniform_real_distribution<float> vd(
                bounds.voltage_threshold_min, bounds.voltage_threshold_max);
            g.voltage_threshold = vd(rng);
        }
    }
    else
    {
        apply_phase00_temporal_coupling(g);
    }
}

thesis::ThesisConfig::AutoencoderConfig to_ae_config(const Genome& g, const std::string& model)
{
    thesis::ThesisConfig::AutoencoderConfig ae;
    ae.model = model;
    ae.encoder_layer_spec = {"linear:" + std::to_string(g.hidden) + ":leaky",
        "linear:" + std::to_string(g.latent) + ":identity"};
    ae.decoder_layer_spec = {
        "linear:" + std::to_string(g.hidden) + ":leaky", "linear:output:identity"};

    if (model == "snn-ae")
    {
        ae.encoding = g.encoding;
        ae.time_steps = g.time_steps;
        ae.voltage_threshold = g.voltage_threshold;
        // phase00-fixed firing-rate regularization for the SNN-AE encoder.
        ae.firing_rate_reg_lambda = 0.5f;
        ae.firing_rate_min = 0.1f;
        ae.firing_rate_max = 0.8f;
    }
    else // ann-ae: non-spiking, always analog/direct single frame.
    {
        ae.encoding = "direct";
        ae.time_steps = 1;
        ae.voltage_threshold = 1.0f;
    }
    return ae;
}

long estimated_params(const Genome& g)
{
    const long in = kAeInputFeatures;
    const long h = g.hidden;
    const long l = g.latent;
    // encoder: (in·h + h) + (h·l + l);  decoder: (l·h + h) + (h·in + in)
    return (in * h + h) + (h * l + l) + (l * h + h) + (h * in + in);
}

long encoder_macs_per_frame(const Genome& g)
{
    return static_cast<long>(kAeInputFeatures) * g.hidden + static_cast<long>(g.hidden) * g.latent;
}

long inference_cost_proxy(const Genome& g)
{
    return encoder_macs_per_frame(g) * std::max(1, g.time_steps);
}

} // namespace pga
