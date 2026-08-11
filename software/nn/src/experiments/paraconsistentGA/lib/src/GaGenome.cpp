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

int rand_width(std::mt19937& rng, const GenomeBounds& b)
{
    std::uniform_int_distribution<int> d(b.min_width, b.max_width);
    return d(rng);
}
} // namespace

void repair_widths(Genome& g, const GenomeBounds& bounds)
{
    auto& w = g.encoder_widths;

    if (w.empty()) w.push_back(bounds.max_width); // seed; the strict-decrease pass fixes it

    // Clamp each width into [min_width, max_width].
    for (int& x : w) x = std::clamp(x, bounds.min_width, bounds.max_width);

    // Strictly decreasing: sort descending, then walk down forcing each entry below
    // its predecessor. When there is no integer room left (would drop below
    // min_width), truncate — a shorter genome is legal, an equal/rising one is not.
    std::sort(w.begin(), w.end(), std::greater<int>());
    std::vector<int> fixed;
    fixed.reserve(w.size());
    int ceiling = bounds.max_width + 1; // strictly-below target for the first entry
    for (int x : w)
    {
        int v = std::min(x, ceiling - 1);
        if (v < bounds.min_width) break; // no room to keep strictly decreasing
        fixed.push_back(v);
        ceiling = v;
    }
    if (fixed.empty()) fixed.push_back(bounds.min_width);

    // Respect max_layers (min_layers is guaranteed reachable by validate()).
    if (static_cast<int>(fixed.size()) > bounds.max_layers)
        fixed.resize(static_cast<size_t>(bounds.max_layers));

    w = std::move(fixed);
}

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

    std::uniform_int_distribution<int> depth_d(bounds.min_layers, bounds.max_layers);
    const int depth = depth_d(rng);
    g.encoder_widths.clear();
    g.encoder_widths.reserve(static_cast<size_t>(depth));
    for (int i = 0; i < depth; ++i) g.encoder_widths.push_back(rand_width(rng, bounds));
    repair_widths(g, bounds); // sort strictly-decreasing, clamp, bound depth

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

Genome crossover(
    const Genome& a, const Genome& b, std::mt19937& rng, const GenomeBounds& bounds, bool is_snn)
{
    std::bernoulli_distribution coin(0.5);
    Genome c;

    // Child depth inherited from one parent; each position sampled from whichever
    // parent has it (or the other, when only one is long enough).
    const size_t depth = coin(rng) ? a.encoder_widths.size() : b.encoder_widths.size();
    c.encoder_widths.clear();
    c.encoder_widths.reserve(depth);
    for (size_t i = 0; i < depth; ++i)
    {
        const bool a_has = i < a.encoder_widths.size();
        const bool b_has = i < b.encoder_widths.size();
        int w;
        if (a_has && b_has)
            w = coin(rng) ? a.encoder_widths[i] : b.encoder_widths[i];
        else if (a_has)
            w = a.encoder_widths[i];
        else
            w = b.encoder_widths[i];
        c.encoder_widths.push_back(w);
    }
    repair_widths(c, bounds);

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
    std::uniform_int_distribution<int> idx(0, std::max(0, g.depth() - 1));

    // 1) jitter one layer's width
    if (hit(rng) && !g.encoder_widths.empty())
        g.encoder_widths[static_cast<size_t>(idx(rng))] = rand_width(rng, bounds);

    // 2) add a layer (if depth budget allows)
    if (hit(rng) && g.depth() < bounds.max_layers)
        g.encoder_widths.push_back(rand_width(rng, bounds));

    // 3) remove a layer (never below min_layers)
    if (hit(rng) && g.depth() > bounds.min_layers)
    {
        const int r = idx(rng);
        g.encoder_widths.erase(g.encoder_widths.begin() + r);
    }

    repair_widths(g, bounds);

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

DiploidGenome random_diploid(std::mt19937& rng, const GenomeBounds& bounds, bool is_snn)
{
    std::uniform_real_distribution<float> dom(0.0f, 1.0f);
    DiploidGenome d;
    d.hap_a = random_genome(rng, bounds, is_snn);
    d.hap_b = random_genome(rng, bounds, is_snn);
    d.dom_a = dom(rng);
    d.dom_b = dom(rng);
    return d;
}

Gamete meiosis(const DiploidGenome& parent,
    std::mt19937& rng,
    const GenomeBounds& bounds,
    double recomb_prob,
    double mutation_prob,
    bool is_snn)
{
    std::bernoulli_distribution recombine(recomb_prob);
    std::bernoulli_distribution hit(mutation_prob);
    std::bernoulli_distribution coin(0.5);

    Gamete g;
    // Recombine the parent's own two haplotypes, or (no crossover this meiosis) carry
    // one haplotype through unchanged.
    g.haplotype = recombine(rng) ? crossover(parent.hap_a, parent.hap_b, rng, bounds, is_snn)
                                 : (coin(rng) ? parent.hap_a : parent.hap_b);
    mutate(g.haplotype, rng, bounds, mutation_prob, is_snn);

    // The gamete carries one parent-haplotype's dominance value; a mutation can redraw
    // it, which is how a formerly recessive haplotype can become dominant in a child.
    g.dominance = coin(rng) ? parent.dom_a : parent.dom_b;
    if (hit(rng))
    {
        std::uniform_real_distribution<float> dom(0.0f, 1.0f);
        g.dominance = dom(rng);
    }
    return g;
}

DiploidGenome fuse(const Gamete& g1, const Gamete& g2)
{
    DiploidGenome d;
    d.hap_a = g1.haplotype;
    d.dom_a = g1.dominance;
    d.hap_b = g2.haplotype;
    d.dom_b = g2.dominance;
    return d;
}

thesis::ThesisConfig::AutoencoderConfig to_ae_config(
    const Genome& g, const thesis::ThesisConfig::AutoencoderConfig& base)
{
    // Copy the profile's config, then override only what the genome owns. Never build
    // from scratch: anything not re-set here would silently revert to a struct default
    // (this is how `ae_loss_type` was previously lost — see the header comment).
    thesis::ThesisConfig::AutoencoderConfig ae = base;

    // Encoder: every width is a leaky Linear except the latent (last), which is
    // identity. Decoder mirrors the hidden widths in reverse, then projects to output.
    const auto& w = g.encoder_widths;
    ae.encoder_layer_spec.clear();
    for (size_t i = 0; i + 1 < w.size(); ++i)
        ae.encoder_layer_spec.push_back("linear:" + std::to_string(w[i]) + ":leaky");
    ae.encoder_layer_spec.push_back("linear:" + std::to_string(w.back()) + ":identity");

    ae.decoder_layer_spec.clear();
    for (size_t i = w.size() - 1; i-- > 0;)
        ae.decoder_layer_spec.push_back("linear:" + std::to_string(w[i]) + ":leaky");
    ae.decoder_layer_spec.push_back("linear:output:identity");

    if (ae.model == "snn-ae")
    {
        ae.encoding = g.encoding;
        ae.time_steps = g.time_steps;
        ae.voltage_threshold = g.voltage_threshold;
        // firing_rate_* deliberately NOT overridden — the profile owns them.
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
    const auto& w = g.encoder_widths;
    long p = 0;
    // encoder: in -> w0 -> w1 -> ... -> latent  (weights + biases per stage)
    long prev = in;
    for (int width : w)
    {
        p += prev * width + width;
        prev = width;
    }
    // decoder: latent -> w[L-2] -> ... -> w0 -> output(in)
    for (size_t i = w.size() - 1; i-- > 0;)
    {
        p += prev * w[i] + w[i];
        prev = w[i];
    }
    p += prev * in + in; // final projection to output
    return p;
}

long encoder_macs_per_frame(const Genome& g)
{
    const auto& w = g.encoder_widths;
    long macs = 0;
    long prev = kAeInputFeatures;
    for (int width : w)
    {
        macs += prev * width;
        prev = width;
    }
    return macs;
}

long inference_cost_proxy(const Genome& g)
{
    return encoder_macs_per_frame(g) * std::max(1, g.time_steps);
}

} // namespace pga
