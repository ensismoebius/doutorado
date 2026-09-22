#include "../include/Meeting01GaSearch.hpp"

#include <algorithm>
#include <filesystem>
#include <map>
#include <random>
#include <stdexcept>

#include "../include/Meeting01GaCheckpoint.hpp"
#include "ga/Nsga2Core.hpp"

namespace meeting01::ga
{

namespace
{
void rerank(std::vector<Meeting01GaIndividual>& pop)
{
    auto fronts = ::ga::fast_non_dominated_sort(pop);
    for (const auto& f : fronts) ::ga::assign_crowding_distance(pop, f);
}

// Crowded-comparison operator: lower rank wins; tie broken by larger crowding.
bool crowded_less(const Meeting01GaIndividual& a, const Meeting01GaIndividual& b)
{
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowding > b.crowding;
}

// Evaluation cache keyed by genome — identical genomes are trained once.
struct EvalCache
{
    std::map<std::string, Meeting01GaIndividual> table;
    std::vector<Meeting01GaIndividual> history;
    std::string persist_path;

    void preload(const std::vector<Meeting01GaIndividual>& entries)
    {
        for (const auto& ind : entries)
        {
            const std::string key = genome_key(ind.genome);
            if (table.emplace(key, ind).second) history.push_back(ind);
        }
    }

    Meeting01GaIndividual& get(Genome g,
        int gen,
        const meeting01::Meeting01Config& cfg,
        const meeting01::DatasetSplit& split,
        std::uint32_t seed,
        std::size_t idx,
        std::size_t total,
        const EvalCallback& on_eval)
    {
        const std::string key = genome_key(g);
        auto it = table.find(key);
        if (it != table.end()) return it->second;

        Meeting01GaIndividual ind;
        ind.genome = std::move(g);
        ind.born_generation = gen;
        evaluate_individual(ind, cfg, split, seed, idx, total);
        if (on_eval) on_eval(ind);

        if (!persist_path.empty()) append_cache_entry(persist_path, ind);

        auto [ins, _] = table.emplace(key, ind);
        history.push_back(ind);
        return ins->second;
    }
};

// Binary tournament; `exclude` (when >= 0) is never returned, so the second parent of a
// mating can never be the first (no self-mating).
//
// Throws instead of spinning when the exclusion is unsatisfiable: with a population of
// one, the rejection loop below would draw the only index forever, burning a core with
// no output and no error — the worst possible failure on a multi-week cluster job.
// `check_ga` rejects that configuration up front; this is the second line of defence.
int tournament(
    const std::vector<Meeting01GaIndividual>& pop, std::mt19937& rng, int k, int exclude = -1)
{
    if (exclude >= 0 && pop.size() < 2)
    {
        throw std::invalid_argument(
            "Meeting01GaSearch: tournament() cannot exclude an individual from a population "
            "of " +
            std::to_string(pop.size()) +
            "; population_size must be >= 2 whenever generations >= 1.");
    }
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pop.size()) - 1);
    auto draw = [&]
    {
        int c = pick(rng);
        while (c == exclude) c = pick(rng);
        return c;
    };
    int best = draw();
    for (int i = 1; i < k; ++i)
    {
        int c = draw();
        if (crowded_less(pop[c], pop[best])) best = c;
    }
    return best;
}

// μ+λ selection over `combined` (parents+offspring): the best N by (rank, then
// crowding) — textbook NSGA-II elitism, no diversity-reserve quota (paraconsistentGA's
// n_losers is opt-in there and not asked for here).
std::vector<Meeting01GaIndividual> select_next_generation(
    const std::vector<Meeting01GaIndividual>& combined,
    const std::vector<std::vector<int>>& fronts,
    int N)
{
    std::vector<Meeting01GaIndividual> next;
    next.reserve(static_cast<std::size_t>(N));
    for (const auto& front : fronts)
    {
        if (static_cast<int>(next.size()) >= N) break;
        if (static_cast<int>(next.size() + front.size()) <= N)
        {
            for (int idx : front) next.push_back(combined[static_cast<std::size_t>(idx)]);
        }
        else
        {
            std::vector<int> ordered(front);
            std::sort(ordered.begin(),
                ordered.end(),
                [&](int x, int y) { return combined[x].crowding > combined[y].crowding; });
            for (int idx : ordered)
            {
                if (static_cast<int>(next.size()) >= N) break;
                next.push_back(combined[static_cast<std::size_t>(idx)]);
            }
        }
    }
    return next;
}
} // namespace

auto run_ga_search(const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfig& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback& on_eval) -> GaSearchResult
{
    std::mt19937 rng(ga_cfg.seed != 0u ? ga_cfg.seed : base_seed);
    const int N = ga_cfg.population_size;

    EvalCache cache;
    const bool ckpt = !ga_cfg.results_dir.empty();
    if (ckpt)
    {
        std::filesystem::create_directories(ga_cfg.results_dir);
        cache.persist_path = checkpoint_cache_path(ga_cfg.results_dir, ga_cfg.run_tag);
        cache.preload(load_cache_entries(cache.persist_path));
    }

    std::vector<Meeting01GaIndividual> parents;
    int start_gen = 1;

    if (ckpt && state_checkpoint_exists(ga_cfg.results_dir, ga_cfg.run_tag))
    {
        GenerationCheckpoint ck = load_generation_checkpoint(ga_cfg.results_dir, ga_cfg.run_tag);
        rng_from_string(rng, ck.rng_state);
        parents = std::move(ck.parents);
        rerank(parents);
        start_gen = ck.generation + 1;
    }
    else
    {
        parents.reserve(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i)
        {
            Genome g = random_genome(rng, ga_cfg.bounds);
            parents.push_back(cache.get(std::move(g),
                0,
                cfg,
                split,
                base_seed,
                static_cast<std::size_t>(i),
                static_cast<std::size_t>(N),
                on_eval));
        }
        rerank(parents);
        if (ckpt) save_generation_checkpoint(ga_cfg.results_dir, ga_cfg.run_tag, 0, rng, parents);
    }

    std::bernoulli_distribution do_cross(ga_cfg.crossover_prob);
    std::bernoulli_distribution coin(0.5);

    for (int gen = start_gen; gen <= ga_cfg.generations; ++gen)
    {
        std::vector<Meeting01GaIndividual> offspring;
        offspring.reserve(static_cast<std::size_t>(N));
        while (static_cast<int>(offspring.size()) < N)
        {
            const int a = tournament(parents, rng, ga_cfg.tournament_k);
            const int b = tournament(parents, rng, ga_cfg.tournament_k, a);

            Genome child = do_cross(rng)
                               ? crossover(parents[a].genome, parents[b].genome, rng, ga_cfg.bounds)
                               : (coin(rng) ? parents[a].genome : parents[b].genome);
            mutate(child, rng, ga_cfg.bounds, ga_cfg.mutation_prob);

            offspring.push_back(cache.get(std::move(child),
                gen,
                cfg,
                split,
                base_seed,
                offspring.size(),
                static_cast<std::size_t>(N),
                on_eval));
        }

        std::vector<Meeting01GaIndividual> combined = parents;
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        auto cfronts = ::ga::fast_non_dominated_sort(combined);
        for (const auto& front : cfronts) ::ga::assign_crowding_distance(combined, front);

        parents = select_next_generation(combined, cfronts, N);
        rerank(parents);

        if (ckpt && (gen % std::max(1, ga_cfg.checkpoint_every_generations) == 0 ||
                        gen == ga_cfg.generations))
            save_generation_checkpoint(ga_cfg.results_dir, ga_cfg.run_tag, gen, rng, parents);
    }

    GaSearchResult result;
    result.generations_run = ga_cfg.generations;
    result.final_population = parents;
    result.history = cache.history;

    auto ff = ::ga::fast_non_dominated_sort(parents);
    for (const auto& f : ff) ::ga::assign_crowding_distance(parents, f);
    if (!ff.empty())
        for (int idx : ff[0])
            if (parents[static_cast<std::size_t>(idx)].feasible)
                result.pareto_front.push_back(parents[static_cast<std::size_t>(idx)]);

    // Winner's-curse mitigation: re-score the front on extra seeds and replace each
    // member's val_mse with the mean, so the pick reflects expected quality rather than
    // the single luckiest evaluation. Only the front is re-scored, not all 90 genomes.
    if (ga_cfg.winner_seeds > 1)
    {
        for (auto& ind : result.pareto_front)
        {
            double mse_sum = ind.val_mse;
            for (int extra = 1; extra < ga_cfg.winner_seeds; ++extra)
            {
                Meeting01GaIndividual probe;
                probe.genome = ind.genome;
                evaluate_individual(
                    probe, cfg, split, base_seed + static_cast<std::uint32_t>(extra) * 7919u, 0, 0);
                if (on_eval) on_eval(probe);
                mse_sum += probe.val_mse;
            }
            ind.val_mse = static_cast<float>(mse_sum / ga_cfg.winner_seeds);
            if (!ind.objectives.empty()) ind.objectives[0] = ind.val_mse;
        }
    }

    std::sort(result.pareto_front.begin(),
        result.pareto_front.end(),
        [](const Meeting01GaIndividual& a, const Meeting01GaIndividual& b)
        {
            if (a.val_mse != b.val_mse) return a.val_mse < b.val_mse;
            return a.inference_cost < b.inference_cost;
        });

    if (ckpt) remove_checkpoint_artifacts(ga_cfg.results_dir, ga_cfg.run_tag);

    return result;
}

auto pick_winner(const GaSearchResult& result) -> const Meeting01GaIndividual&
{
    if (result.pareto_front.empty())
        throw std::runtime_error(
            "Meeting01GaSearch: pareto_front is empty — should be unreachable (population_size "
            ">= 1 and every individual here is unconditionally feasible)");
    return result.pareto_front.front();
}

} // namespace meeting01::ga
