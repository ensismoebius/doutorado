#include "GaNsga2.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "progress/ProgressManager.hpp"

namespace pga
{

bool constrained_dominates(const Individual& a, const Individual& b)
{
    if (a.feasible != b.feasible) return a.feasible; // feasible dominates infeasible
    if (!a.feasible && !b.feasible)                  // both infeasible: less violation wins
        return a.constraint_violation < b.constraint_violation;

    // Both feasible: standard Pareto on minimized objectives.
    bool strictly_better = false;
    const size_t m = std::min(a.objectives.size(), b.objectives.size());
    for (size_t i = 0; i < m; ++i)
    {
        if (a.objectives[i] > b.objectives[i]) return false;
        if (a.objectives[i] < b.objectives[i]) strictly_better = true;
    }
    return strictly_better;
}

std::vector<std::vector<int>> fast_non_dominated_sort(std::vector<Individual>& pop)
{
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> dominates(n);
    std::vector<int> dominated_count(n, 0);
    std::vector<std::vector<int>> fronts;
    fronts.emplace_back();

    for (int p = 0; p < n; ++p)
    {
        for (int q = 0; q < n; ++q)
        {
            if (p == q) continue;
            if (constrained_dominates(pop[p], pop[q]))
                dominates[p].push_back(q);
            else if (constrained_dominates(pop[q], pop[p]))
                ++dominated_count[p];
        }
        if (dominated_count[p] == 0)
        {
            pop[p].rank = 0;
            fronts[0].push_back(p);
        }
    }

    int fi = 0;
    while (!fronts[fi].empty())
    {
        std::vector<int> next;
        for (int p : fronts[fi])
            for (int q : dominates[p])
                if (--dominated_count[q] == 0)
                {
                    pop[q].rank = fi + 1;
                    next.push_back(q);
                }
        ++fi;
        fronts.push_back(std::move(next));
    }
    fronts.pop_back(); // last pushed front is empty
    return fronts;
}

void assign_crowding_distance(std::vector<Individual>& pop, const std::vector<int>& front)
{
    const size_t l = front.size();
    for (int idx : front) pop[idx].crowding = 0.0;
    if (l == 0) return;
    if (l <= 2)
    {
        for (int idx : front) pop[idx].crowding = std::numeric_limits<double>::infinity();
        return;
    }

    const size_t n_obj = pop[front[0]].objectives.size();
    for (size_t m = 0; m < n_obj; ++m)
    {
        std::vector<int> order(front);
        std::sort(order.begin(),
            order.end(),
            [&](int a, int b) { return pop[a].objectives[m] < pop[b].objectives[m]; });

        pop[order.front()].crowding = std::numeric_limits<double>::infinity();
        pop[order.back()].crowding = std::numeric_limits<double>::infinity();

        const double lo = pop[order.front()].objectives[m];
        const double hi = pop[order.back()].objectives[m];
        const double range = hi - lo;
        if (range <= 0.0) continue;

        for (size_t i = 1; i + 1 < l; ++i)
        {
            if (pop[order[i]].crowding == std::numeric_limits<double>::infinity()) continue;
            pop[order[i]].crowding +=
                (pop[order[i + 1]].objectives[m] - pop[order[i - 1]].objectives[m]) / range;
        }
    }
}

namespace
{
// Crowded-comparison operator: lower rank wins; tie broken by larger crowding.
bool crowded_less(const Individual& a, const Individual& b)
{
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowding > b.crowding;
}

std::string genome_key(const Genome& g)
{
    std::ostringstream s;
    s << g.hidden << ':' << g.latent << ':' << g.encoding << ':' << g.time_steps << ':'
      << g.voltage_threshold;
    return s.str();
}

// Evaluation cache: identical genomes are trained once. Keeps the whole run
// deterministic AND cheaper (a genome that survives generations is not retrained).
struct EvalCache
{
    std::map<std::string, Individual> table;
    std::vector<Individual> history; // one entry per distinct genome, in discovery order

    Individual& get(Genome g,
        int generation,
        const thesis::ThesisDatasetView& view,
        const GaConfig& cfg,
        const EvalCallback& on_eval)
    {
        const std::string key = genome_key(g);
        auto it = table.find(key);
        if (it != table.end()) return it->second;

        Individual ind;
        ind.genome = std::move(g);
        ind.born_generation = generation;
        evaluate_individual(ind, view, cfg);
        if (on_eval) on_eval(ind);

        auto [ins, _] = table.emplace(key, ind);
        history.push_back(ind);
        return ins->second;
    }
};

int tournament(const std::vector<Individual>& pop, std::mt19937& rng, int k)
{
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pop.size()) - 1);
    int best = pick(rng);
    for (int i = 1; i < k; ++i)
    {
        int c = pick(rng);
        if (crowded_less(pop[c], pop[best])) best = c;
    }
    return best;
}
} // namespace

GaResult run_nsga2(
    const thesis::ThesisDatasetView& view, const GaConfig& cfg, const EvalCallback& on_eval)
{
    auto& pm = nn::progress::ProgressManager::instance();
    std::mt19937 rng(cfg.ga.seed);
    const bool is_snn = cfg.is_snn_population();
    const int N = cfg.ga.population_size;

    EvalCache cache;

    // ── Initial population ───────────────────────────────────────────────────
    pm.log("[PGA] Generation 0: initializing + evaluating " + std::to_string(N) + " individuals (" +
           cfg.base.feature_extraction.autoencoder.model + " / " + cfg.base.dataset.modality + ")");
    std::vector<Individual> parents;
    parents.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        parents.push_back(
            cache.get(random_genome(rng, cfg.ga.bounds, is_snn), 0, view, cfg, on_eval));

    auto fronts = fast_non_dominated_sort(parents);
    for (const auto& f : fronts) assign_crowding_distance(parents, f);

    // ── Generational loop ────────────────────────────────────────────────────
    std::bernoulli_distribution do_crossover(cfg.ga.crossover_prob);
    for (int gen = 1; gen <= cfg.ga.generations; ++gen)
    {
        std::vector<Individual> offspring;
        offspring.reserve(static_cast<size_t>(N));
        while (static_cast<int>(offspring.size()) < N)
        {
            const int a = tournament(parents, rng, cfg.ga.tournament_k);
            const int b = tournament(parents, rng, cfg.ga.tournament_k);
            Genome child = do_crossover(rng)
                               ? crossover(parents[a].genome, parents[b].genome, rng, is_snn)
                               : parents[a].genome;
            mutate(child, rng, cfg.ga.bounds, cfg.ga.mutation_prob, is_snn);
            offspring.push_back(cache.get(std::move(child), gen, view, cfg, on_eval));
        }

        // μ+λ selection: combine parents and offspring, sort, take the best N.
        std::vector<Individual> combined;
        combined.reserve(parents.size() + offspring.size());
        combined.insert(combined.end(), parents.begin(), parents.end());
        combined.insert(combined.end(), offspring.begin(), offspring.end());

        auto cfronts = fast_non_dominated_sort(combined);
        std::vector<Individual> next;
        next.reserve(static_cast<size_t>(N));
        for (const auto& front : cfronts)
        {
            assign_crowding_distance(combined, front);
            if (static_cast<int>(next.size() + front.size()) <= N)
            {
                for (int idx : front) next.push_back(combined[idx]);
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
                    next.push_back(combined[idx]);
                }
                break;
            }
            if (static_cast<int>(next.size()) >= N) break;
        }
        parents = std::move(next);
        // Refresh rank/crowding for the surviving population (used by next tournament).
        auto pf = fast_non_dominated_sort(parents);
        for (const auto& f : pf) assign_crowding_distance(parents, f);

        int n_feasible = 0;
        double best_dpen = std::numeric_limits<double>::max();
        for (const auto& ind : parents)
            if (ind.feasible)
            {
                ++n_feasible;
                best_dpen = std::min(best_dpen, ind.d_penalized_mean);
            }
        std::ostringstream oss;
        oss << "[PGA] Generation " << gen << "/" << cfg.ga.generations << ": " << n_feasible << "/"
            << N << " feasible, best feasible d_penalized="
            << (n_feasible ? best_dpen : std::numeric_limits<double>::quiet_NaN()) << ", "
            << cache.table.size() << " distinct genomes evaluated";
        pm.log(oss.str());
    }

    // ── Assemble result ──────────────────────────────────────────────────────
    GaResult result;
    result.generations_run = cfg.ga.generations;
    result.final_population = parents;
    result.history = cache.history;

    // Final Pareto front = feasible, rank-0 individuals of the last population.
    auto ff = fast_non_dominated_sort(parents);
    for (const auto& f : ff) assign_crowding_distance(parents, f);
    if (!ff.empty())
        for (int idx : ff[0])
            if (parents[idx].feasible) result.pareto_front.push_back(parents[idx]);

    // Deterministic front ordering: by inference cost then d_penalized.
    std::sort(result.pareto_front.begin(),
        result.pareto_front.end(),
        [](const Individual& a, const Individual& b)
        {
            if (a.inference_cost != b.inference_cost) return a.inference_cost < b.inference_cost;
            return a.d_penalized_mean < b.d_penalized_mean;
        });

    return result;
}

} // namespace pga
