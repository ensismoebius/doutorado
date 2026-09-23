#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01DatasetSplit.hpp"
#include "Meeting01GaCheckpoint.hpp"
#include "Meeting01GaFitness.hpp"
#include "Meeting01GaGenome.hpp"
#include "ga/Nsga2Core.hpp"

namespace meeting01::ga
{

// Generalized 2026-09-22 from an SNN-only driver into a family-agnostic template: the
// generational loop (tournament selection, μ+λ elitism, checkpointing) is IDENTICAL
// across SNN/LSTM/GRU/Transformer — only the genome shape and how one individual is
// trained differ, and those already live in per-family files (Meeting01*GaGenome.*,
// Meeting01*GaFitness.*). `Ind` names the individual type (its `genome_type` alias
// names the genome); `Bounds` names that genome's bounds struct. Every genome-specific
// operation this header calls (`random_genome`, `crossover`, `mutate`, `genome_key`,
// `evaluate_individual`) is called UNQUALIFIED and resolved via ADL on the concrete
// `Bounds`/`Ind` types at each instantiation — the same idiom `std::swap` relies on,
// and the reason this file never needs to name "SNN" or "LSTM" anywhere in its body.

template <typename Bounds>
struct GaSearchConfigT
{
    int population_size = 10;
    int generations = 8;
    double crossover_prob = 0.9;
    double mutation_prob = 0.2;
    int tournament_k = 2;
    unsigned int seed = 0; // 0 -> derive from base_seed passed to run_ga_search
    // How many seeds each final Pareto-front member is scored on before the winner is
    // picked. The search itself scores one seed per genome (cheap), which means the best
    // of a noisy population is partly selected on seed luck — the winner's curse.
    // Re-scoring only the front costs |front| * (winner_seeds - 1) extra trainings and
    // makes the published architecture a mean over seeds rather than a lucky draw. 1
    // disables it.
    int winner_seeds = 3;
    Bounds bounds{};

    // Checkpointing (two-layer, Meeting01GaCheckpoint.hpp). Empty results_dir disables it.
    std::string results_dir;
    std::string run_tag;
    int checkpoint_every_generations = 1;
};
using GaSearchConfig = GaSearchConfigT<GenomeBounds>;

template <typename Ind>
struct GaSearchResultT
{
    std::vector<Ind> final_population;
    // Feasible rank-0 individuals of the final population, sorted by (val_mse, then
    // inference_cost) — reversed from paraconsistentGA's (cost, then quality) ordering
    // because meeting01's primary published metric is reconstruction quality, not cost.
    std::vector<Ind> pareto_front;
    std::vector<Ind> history; // every distinct genome ever evaluated
    int generations_run = 0;
};
using GaSearchResult = GaSearchResultT<Meeting01GaIndividual>;

template <typename Ind>
using EvalCallback = std::function<void(const Ind&)>;

namespace detail
{
template <typename Ind>
void rerank(std::vector<Ind>& pop)
{
    auto fronts = ::ga::fast_non_dominated_sort(pop);
    for (const auto& f : fronts) ::ga::assign_crowding_distance(pop, f);
}

// Crowded-comparison operator: lower rank wins; tie broken by larger crowding.
template <typename Ind>
bool crowded_less(const Ind& a, const Ind& b)
{
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowding > b.crowding;
}

// Evaluation cache keyed by genome — identical genomes are trained once.
template <typename Ind, typename Bounds>
struct EvalCache
{
    std::map<std::string, Ind> table;
    std::vector<Ind> history;
    std::string persist_path;

    void preload(const std::vector<Ind>& entries)
    {
        for (const auto& ind : entries)
        {
            const std::string key = genome_key(ind.genome);
            if (table.emplace(key, ind).second) history.push_back(ind);
        }
    }

    Ind& get(typename Ind::genome_type g,
        int gen,
        const meeting01::Meeting01Config& cfg,
        const meeting01::DatasetSplit& split,
        std::uint32_t seed,
        std::size_t idx,
        std::size_t total,
        const EvalCallback<Ind>& on_eval)
    {
        const std::string key = genome_key(g);
        auto it = table.find(key);
        if (it != table.end()) return it->second;

        Ind ind;
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
template <typename Ind>
int tournament(const std::vector<Ind>& pop, std::mt19937& rng, int k, int exclude = -1)
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
        if (crowded_less(pop[static_cast<std::size_t>(c)], pop[static_cast<std::size_t>(best)]))
            best = c;
    }
    return best;
}

// μ+λ selection over `combined` (parents+offspring): the best N by (rank, then
// crowding) — textbook NSGA-II elitism, no diversity-reserve quota.
template <typename Ind>
std::vector<Ind> select_next_generation(
    const std::vector<Ind>& combined, const std::vector<std::vector<int>>& fronts, int N)
{
    std::vector<Ind> next;
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
} // namespace detail

// Runs NSGA-II (haploid — no diploid genetics; see Meeting01GaFitness.hpp for why no
// feasibility constraint is modeled either) over `Ind::genome_type` for one (dataset,
// fold, run_id). `split` is loaded once by the caller. Checkpointed via
// Meeting01GaCheckpoint.hpp iff ga_cfg.results_dir is non-empty.
template <typename Ind, typename Bounds>
auto run_ga_search(const meeting01::Meeting01Config& cfg,
    const meeting01::DatasetSplit& split,
    const GaSearchConfigT<Bounds>& ga_cfg,
    std::uint32_t base_seed,
    const EvalCallback<Ind>& on_eval = nullptr) -> GaSearchResultT<Ind>
{
    std::mt19937 rng(ga_cfg.seed != 0u ? ga_cfg.seed : base_seed);
    const int N = ga_cfg.population_size;

    detail::EvalCache<Ind, Bounds> cache;
    const bool ckpt = !ga_cfg.results_dir.empty();
    if (ckpt)
    {
        std::filesystem::create_directories(ga_cfg.results_dir);
        cache.persist_path = checkpoint_cache_path(ga_cfg.results_dir, ga_cfg.run_tag);
        cache.preload(load_cache_entries<Ind>(cache.persist_path));
    }

    std::vector<Ind> parents;
    int start_gen = 1;

    if (ckpt && state_checkpoint_exists(ga_cfg.results_dir, ga_cfg.run_tag))
    {
        GenerationCheckpoint<Ind> ck =
            load_generation_checkpoint<Ind>(ga_cfg.results_dir, ga_cfg.run_tag);
        rng_from_string(rng, ck.rng_state);
        parents = std::move(ck.parents);
        detail::rerank(parents);
        start_gen = ck.generation + 1;
    }
    else
    {
        parents.reserve(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i)
        {
            typename Ind::genome_type g = random_genome(rng, ga_cfg.bounds);
            parents.push_back(cache.get(std::move(g),
                0,
                cfg,
                split,
                base_seed,
                static_cast<std::size_t>(i),
                static_cast<std::size_t>(N),
                on_eval));
        }
        detail::rerank(parents);
        if (ckpt) save_generation_checkpoint(ga_cfg.results_dir, ga_cfg.run_tag, 0, rng, parents);
    }

    std::bernoulli_distribution do_cross(ga_cfg.crossover_prob);
    std::bernoulli_distribution coin(0.5);

    for (int gen = start_gen; gen <= ga_cfg.generations; ++gen)
    {
        std::vector<Ind> offspring;
        offspring.reserve(static_cast<std::size_t>(N));
        while (static_cast<int>(offspring.size()) < N)
        {
            const int a = detail::tournament(parents, rng, ga_cfg.tournament_k);
            const int b = detail::tournament(parents, rng, ga_cfg.tournament_k, a);

            typename Ind::genome_type child =
                do_cross(rng) ? crossover(parents[static_cast<std::size_t>(a)].genome,
                                    parents[static_cast<std::size_t>(b)].genome,
                                    rng,
                                    ga_cfg.bounds)
                              : (coin(rng) ? parents[static_cast<std::size_t>(a)].genome
                                           : parents[static_cast<std::size_t>(b)].genome);
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

        std::vector<Ind> combined = parents;
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        auto cfronts = ::ga::fast_non_dominated_sort(combined);
        for (const auto& front : cfronts) ::ga::assign_crowding_distance(combined, front);

        parents = detail::select_next_generation(combined, cfronts, N);
        detail::rerank(parents);

        if (ckpt && (gen % std::max(1, ga_cfg.checkpoint_every_generations) == 0 ||
                        gen == ga_cfg.generations))
            save_generation_checkpoint(ga_cfg.results_dir, ga_cfg.run_tag, gen, rng, parents);
    }

    GaSearchResultT<Ind> result;
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
    // the single luckiest evaluation. Only the front is re-scored, not the whole population.
    if (ga_cfg.winner_seeds > 1)
    {
        for (auto& ind : result.pareto_front)
        {
            double mse_sum = ind.val_mse;
            for (int extra = 1; extra < ga_cfg.winner_seeds; ++extra)
            {
                Ind probe;
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
        [](const Ind& a, const Ind& b)
        {
            if (a.val_mse != b.val_mse) return a.val_mse < b.val_mse;
            return a.inference_cost < b.inference_cost;
        });

    if (ckpt) remove_checkpoint_artifacts(ga_cfg.results_dir, ga_cfg.run_tag);

    return result;
}

// The individual the final retrain+test tail should use: result.pareto_front.front()
// (already ordered by val_mse then cost — reproduces today's min_element-by-val_mse
// grid-selection semantics as a special case). Throws if pareto_front is empty, which
// should be unreachable: population_size >= 1 and every individual here is
// unconditionally feasible.
template <typename Ind>
auto pick_winner(const GaSearchResultT<Ind>& result) -> const Ind&
{
    if (result.pareto_front.empty())
        throw std::runtime_error(
            "Meeting01GaSearch: pareto_front is empty — should be unreachable (population_size "
            ">= 1 and every individual here is unconditionally feasible)");
    return result.pareto_front.front();
}

} // namespace meeting01::ga
