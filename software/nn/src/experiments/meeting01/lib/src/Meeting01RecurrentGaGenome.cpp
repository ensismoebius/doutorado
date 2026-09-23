#include "../include/Meeting01RecurrentGaGenome.hpp"

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
            "Meeting01RecurrentGa: choice list is empty — RecurrentGenomeBounds must "
            "carry at least one legal encoding value");
    std::uniform_int_distribution<std::size_t> d(0, choices.size() - 1);
    return choices[d(rng)];
}
} // namespace

void repair_recurrent(RecurrentGenome& g, const RecurrentGenomeBounds& bounds)
{
    g.hidden_size = std::clamp(g.hidden_size, bounds.min_hidden, bounds.max_hidden);
    g.num_layers = std::clamp(g.num_layers, bounds.min_layers, bounds.max_layers);
}

RecurrentGenome random_genome(std::mt19937& rng, const RecurrentGenomeBounds& bounds)
{
    RecurrentGenome g;
    std::uniform_int_distribution<int> hidden_d(bounds.min_hidden, bounds.max_hidden);
    std::uniform_int_distribution<int> layers_d(bounds.min_layers, bounds.max_layers);
    g.hidden_size = hidden_d(rng);
    g.num_layers = layers_d(rng);
    g.encoding = pick(rng, bounds.encoding_choices);
    repair_recurrent(g, bounds);
    return g;
}

RecurrentGenome crossover(const RecurrentGenome& a,
    const RecurrentGenome& b,
    std::mt19937& rng,
    const RecurrentGenomeBounds& bounds)
{
    std::bernoulli_distribution coin(0.5);
    RecurrentGenome c;
    c.hidden_size = coin(rng) ? a.hidden_size : b.hidden_size;
    c.num_layers = coin(rng) ? a.num_layers : b.num_layers;
    c.encoding = coin(rng) ? a.encoding : b.encoding;
    repair_recurrent(c, bounds);
    return c;
}

void mutate(RecurrentGenome& g, std::mt19937& rng, const RecurrentGenomeBounds& bounds, double prob)
{
    std::bernoulli_distribution hit(prob);
    std::uniform_int_distribution<int> hidden_d(bounds.min_hidden, bounds.max_hidden);
    std::uniform_int_distribution<int> layers_d(bounds.min_layers, bounds.max_layers);

    if (hit(rng)) g.hidden_size = hidden_d(rng);
    if (hit(rng)) g.num_layers = layers_d(rng);
    if (hit(rng)) g.encoding = pick(rng, bounds.encoding_choices);

    repair_recurrent(g, bounds);
}

auto to_lstm_cfg(const RecurrentGenome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::lstm::LSTMAutoencoderConfig
{
    return meeting01::make_lstm_cfg(cfg, g.hidden_size, g.num_layers);
}

auto to_gru_cfg(const RecurrentGenome& g, const meeting01::Meeting01Config& cfg)
    -> nn::models::gru::GRUAutoencoderConfig
{
    return meeting01::make_gru_cfg(cfg, g.hidden_size, g.num_layers);
}

auto genome_key(const RecurrentGenome& g) -> std::string
{
    std::ostringstream s;
    s << g.hidden_size << '|' << g.num_layers << '|' << g.encoding;
    return s.str();
}

auto genome_to_json(const RecurrentGenome& g) -> nlohmann::json
{
    return {{"hidden_size", g.hidden_size}, {"num_layers", g.num_layers}, {"encoding", g.encoding}};
}

void genome_from_json(const nlohmann::json& j, RecurrentGenome& out)
{
    out.hidden_size = j.at("hidden_size").get<int>();
    out.num_layers = j.at("num_layers").get<int>();
    out.encoding = j.at("encoding").get<std::string>();
}

} // namespace meeting01::ga
