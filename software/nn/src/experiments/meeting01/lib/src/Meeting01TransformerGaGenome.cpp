#include "../include/Meeting01TransformerGaGenome.hpp"

#include <algorithm>
#include <cmath>
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
            "Meeting01TransformerGa: choice list is empty — TransformerGenomeBounds must "
            "carry at least one legal value");
    std::uniform_int_distribution<std::size_t> d(0, choices.size() - 1);
    return choices[d(rng)];
}

// Nearest entry in `choices` to `value` (ties broken toward the first match).
int nearest(int value, const std::vector<int>& choices)
{
    int best = choices.front();
    int best_diff = std::abs(value - best);
    for (int c : choices)
    {
        const int diff = std::abs(value - c);
        if (diff < best_diff)
        {
            best = c;
            best_diff = diff;
        }
    }
    return best;
}
} // namespace

void repair_transformer(TransformerGenome& g, const TransformerGenomeBounds& bounds)
{
    if (bounds.head_choices.empty())
        throw std::invalid_argument(
            "Meeting01TransformerGa: TransformerGenomeBounds.head_choices is empty — "
            "at least one legal head count is required");

    // n_heads must be one of the legal counts.
    if (std::find(bounds.head_choices.begin(), bounds.head_choices.end(), g.n_heads) ==
        bounds.head_choices.end())
        g.n_heads = nearest(g.n_heads, bounds.head_choices);

    g.n_layers = std::clamp(g.n_layers, bounds.min_layers, bounds.max_layers);
    g.d_ff = std::clamp(g.d_ff, bounds.min_d_ff, bounds.max_d_ff);

    // d_model must be a positive multiple of n_heads inside [min_d_model, max_d_model]:
    // each attention head gets d_model / n_heads dimensions, so a non-multiple is not a
    // degraded model, it is a model the layer refuses to construct. Round the current
    // d_model to the nearest legal multiple rather than just clamping, so mutation/
    // crossover jitter close to a bound doesn't get thrown away.
    const int lo_k = std::max(1, (bounds.min_d_model + g.n_heads - 1) / g.n_heads); // ceil
    const int hi_k = std::max(lo_k, bounds.max_d_model / g.n_heads);                // floor
    const int k = std::clamp(std::max(1, g.d_model / std::max(1, g.n_heads)), lo_k, hi_k);
    g.d_model = k * g.n_heads;
}

TransformerGenome random_genome(std::mt19937& rng, const TransformerGenomeBounds& bounds)
{
    TransformerGenome g;
    g.n_heads = pick(rng, bounds.head_choices);

    std::uniform_int_distribution<int> d_model_d(bounds.min_d_model, bounds.max_d_model);
    g.d_model = d_model_d(rng);

    std::uniform_int_distribution<int> layers_d(bounds.min_layers, bounds.max_layers);
    g.n_layers = layers_d(rng);

    std::uniform_int_distribution<int> d_ff_d(bounds.min_d_ff, bounds.max_d_ff);
    g.d_ff = d_ff_d(rng);

    g.encoding = pick(rng, bounds.encoding_choices);
    repair_transformer(g, bounds);
    return g;
}

TransformerGenome crossover(const TransformerGenome& a,
    const TransformerGenome& b,
    std::mt19937& rng,
    const TransformerGenomeBounds& bounds)
{
    std::bernoulli_distribution coin(0.5);
    TransformerGenome c;
    // n_heads decided first: repair_transformer below reconciles d_model against
    // WHICHEVER n_heads wins the coin flip, so inheriting them independently (rather
    // than always taking both from the same parent) still yields a legal child.
    c.n_heads = coin(rng) ? a.n_heads : b.n_heads;
    c.d_model = coin(rng) ? a.d_model : b.d_model;
    c.n_layers = coin(rng) ? a.n_layers : b.n_layers;
    c.d_ff = coin(rng) ? a.d_ff : b.d_ff;
    c.encoding = coin(rng) ? a.encoding : b.encoding;
    repair_transformer(c, bounds);
    return c;
}

void mutate(
    TransformerGenome& g, std::mt19937& rng, const TransformerGenomeBounds& bounds, double prob)
{
    std::bernoulli_distribution hit(prob);

    if (hit(rng)) g.n_heads = pick(rng, bounds.head_choices);
    if (hit(rng))
    {
        std::uniform_int_distribution<int> d_model_d(bounds.min_d_model, bounds.max_d_model);
        g.d_model = d_model_d(rng);
    }
    if (hit(rng))
    {
        std::uniform_int_distribution<int> layers_d(bounds.min_layers, bounds.max_layers);
        g.n_layers = layers_d(rng);
    }
    if (hit(rng))
    {
        std::uniform_int_distribution<int> d_ff_d(bounds.min_d_ff, bounds.max_d_ff);
        g.d_ff = d_ff_d(rng);
    }
    if (hit(rng)) g.encoding = pick(rng, bounds.encoding_choices);

    repair_transformer(g, bounds);
}

auto to_transformer_cfg(const TransformerGenome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::transformer::TransformerAutoencoderConfig
{
    return meeting01::make_transformer_cfg(cfg, g.d_model, g.n_heads, g.n_layers, g.d_ff);
}

auto genome_key(const TransformerGenome& g) -> std::string
{
    std::ostringstream s;
    s << g.d_model << '|' << g.n_heads << '|' << g.n_layers << '|' << g.d_ff << '|' << g.encoding;
    return s.str();
}

auto genome_to_json(const TransformerGenome& g) -> nlohmann::json
{
    return {{"d_model", g.d_model},
        {"n_heads", g.n_heads},
        {"n_layers", g.n_layers},
        {"d_ff", g.d_ff},
        {"encoding", g.encoding}};
}

void genome_from_json(const nlohmann::json& j, TransformerGenome& out)
{
    out.d_model = j.at("d_model").get<int>();
    out.n_heads = j.at("n_heads").get<int>();
    out.n_layers = j.at("n_layers").get<int>();
    out.d_ff = j.at("d_ff").get<int>();
    out.encoding = j.at("encoding").get<std::string>();
}

} // namespace meeting01::ga
