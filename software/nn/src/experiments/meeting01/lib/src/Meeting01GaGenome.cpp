#include "../include/Meeting01GaGenome.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "../include/Meeting01Training.hpp"

namespace meeting01::ga
{

namespace
{
template <typename T>
T pick(std::mt19937& rng, const std::vector<T>& choices)
{
    if (choices.empty())
        throw std::invalid_argument(
            "Meeting01Ga: choice list is empty — GenomeBounds must "
            "carry at least one legal encoding/architecture value");
    std::uniform_int_distribution<std::size_t> d(0, choices.size() - 1);
    return choices[d(rng)];
}

int rand_width(std::mt19937& rng, const GenomeBounds& b)
{
    std::uniform_int_distribution<int> d(b.min_width, b.max_width);
    return d(rng);
}

float rand_in(std::mt19937& rng, float lo, float hi)
{
    std::uniform_real_distribution<float> d(lo, hi);
    return d(rng);
}
} // namespace

void repair_widths(Genome& g, const GenomeBounds& bounds)
{
    auto& w = g.encoder_widths;

    if (w.empty()) w.push_back(bounds.max_width); // seed; the strict-decrease pass fixes it

    // The bottleneck (encoder_widths.back(), what build_snn_decoder actually receives
    // as its first layer's expected input via cfg.model.latent_dim) is fixed, never a
    // gene — every HIDDEN width must sit strictly above it, or the strictly-decreasing
    // sequence couldn't terminate at bounds.latent_dim at all. Before this, a genome's
    // last width was just another free draw in [min_width, max_width], decoupled from
    // cfg.model.latent_dim: build_snn_decoder always builds its first Linear expecting
    // exactly cfg.latent_size in features, so any genome whose smallest width landed
    // away from cfg.model.latent_dim crashed the very first real forward pass with a
    // "Linear layer forward: input features (…) do not match expected in_features (…)"
    // — never caught by the existing genome-repair/checkpoint tests, which never
    // construct-and-run the actual network.
    const int hidden_min = std::max(bounds.min_width, bounds.latent_dim + 1);

    for (int& x : w) x = std::clamp(x, hidden_min, bounds.max_width);

    std::sort(w.begin(), w.end(), std::greater<int>());
    std::vector<int> fixed;
    fixed.reserve(w.size());
    int ceiling = bounds.max_width + 1;
    for (int x : w)
    {
        int v = std::min(x, ceiling - 1);
        if (v < hidden_min) break;
        fixed.push_back(v);
        ceiling = v;
    }

    const int hidden_cap = std::max(0, bounds.max_layers - 1);
    if (static_cast<int>(fixed.size()) > hidden_cap)
        fixed.resize(static_cast<std::size_t>(hidden_cap));

    fixed.push_back(bounds.latent_dim);
    w = std::move(fixed);
}

Genome random_genome(std::mt19937& rng, const GenomeBounds& bounds)
{
    Genome g;

    std::uniform_int_distribution<int> depth_d(bounds.min_layers, bounds.max_layers);
    const int depth = depth_d(rng);
    g.encoder_widths.clear();
    g.encoder_widths.reserve(static_cast<std::size_t>(depth));
    for (int i = 0; i < depth; ++i) g.encoder_widths.push_back(rand_width(rng, bounds));
    repair_widths(g, bounds);

    g.encoding = pick(rng, bounds.encoding_choices);
    g.architecture = pick(rng, bounds.architecture_choices);
    g.voltage_threshold = rand_in(rng, bounds.voltage_threshold_min, bounds.voltage_threshold_max);
    g.alpha = rand_in(rng, bounds.alpha_min, bounds.alpha_max);
    return g;
}

Genome crossover(const Genome& a, const Genome& b, std::mt19937& rng, const GenomeBounds& bounds)
{
    std::bernoulli_distribution coin(0.5);
    Genome c;

    const std::size_t depth = coin(rng) ? a.encoder_widths.size() : b.encoder_widths.size();
    c.encoder_widths.clear();
    c.encoder_widths.reserve(depth);
    for (std::size_t i = 0; i < depth; ++i)
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

    c.encoding = coin(rng) ? a.encoding : b.encoding;
    c.architecture = coin(rng) ? a.architecture : b.architecture;
    c.voltage_threshold = 0.5f * (a.voltage_threshold + b.voltage_threshold);
    c.alpha = 0.5f * (a.alpha + b.alpha);
    return c;
}

void mutate(Genome& g, std::mt19937& rng, const GenomeBounds& bounds, double prob)
{
    std::bernoulli_distribution hit(prob);
    std::uniform_int_distribution<int> idx(0, std::max(0, g.depth() - 1));

    // 1) jitter one layer's width
    if (hit(rng) && !g.encoder_widths.empty())
        g.encoder_widths[static_cast<std::size_t>(idx(rng))] = rand_width(rng, bounds);

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

    // 4) categorical genes
    if (hit(rng)) g.encoding = pick(rng, bounds.encoding_choices);
    if (hit(rng)) g.architecture = pick(rng, bounds.architecture_choices);

    // 5) continuous genes
    if (hit(rng))
        g.voltage_threshold =
            rand_in(rng, bounds.voltage_threshold_min, bounds.voltage_threshold_max);
    if (hit(rng)) g.alpha = rand_in(rng, bounds.alpha_min, bounds.alpha_max);
}

auto to_ae_config(const Genome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::autoencoder::AutoencoderConfig
{
    return meeting01::make_snn_cfg(cfg, g.alpha, g.voltage_threshold, g.encoder_widths);
}

auto genome_key(const Genome& g) -> std::string
{
    std::ostringstream s;
    for (int w : g.encoder_widths) s << w << ',';
    s << '|' << g.encoding << '|' << g.architecture << '|' << g.voltage_threshold << '|' << g.alpha;
    return s.str();
}

auto genome_to_json(const Genome& g) -> nlohmann::json
{
    return {{"encoder_widths", g.encoder_widths},
        {"encoding", g.encoding},
        {"architecture", g.architecture},
        {"voltage_threshold", g.voltage_threshold},
        {"alpha", g.alpha}};
}

void genome_from_json(const nlohmann::json& j, Genome& out)
{
    out.encoder_widths = j.at("encoder_widths").get<std::vector<int>>();
    out.encoding = j.at("encoding").get<std::string>();
    out.architecture = j.at("architecture").get<std::string>();
    out.voltage_threshold = j.at("voltage_threshold").get<float>();
    out.alpha = j.at("alpha").get<float>();
}

} // namespace meeting01::ga
