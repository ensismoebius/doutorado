#include "GaNsga2.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>

#include "GaCheckpoint.hpp"
#include "GaModelSnapshot.hpp"
#include "ThesisOutput.hpp" // thesis::ensure_dir
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
    for (int w : g.encoder_widths) s << w << ',';
    s << '|' << g.encoding << ':' << g.time_steps << ':' << g.voltage_threshold;
    return s.str();
}

// Evaluation cache: identical genomes are trained once. Keeps the whole run
// deterministic AND cheaper (a genome that survives generations is not retrained).
struct EvalCache
{
    std::map<std::string, Individual> table;
    std::vector<Individual> history; // one entry per distinct genome, in discovery order
    std::string persist_path;        // non-empty ⇒ append each new eval as a JSONL line

    // Best-of-run weight snapshot: non-empty ⇒ persist a running best-so-far to disk.
    // Updated live so the file is guaranteed to exist by the time the run ends, without
    // ever retraining anything (see GaModelSnapshot.hpp).
    std::string weights_path;
    double best_d_penalized = std::numeric_limits<double>::max();

    // Replay a persisted cache (layer 1): every previously-trained genome becomes a hit, so
    // a resumed run retrains nothing. Discovery order is preserved for the final CSV. Also
    // recovers the weight-snapshot threshold: the .npz on disk from before the crash
    // already holds SOME best-so-far model, so only the numeric bar needs restoring —
    // never the tensors themselves (those were always transient, see Individual header).
    void preload(const std::vector<Individual>& entries)
    {
        for (const auto& ind : entries)
        {
            const std::string key = genome_key(ind.genome);
            if (table.emplace(key, ind).second) history.push_back(ind);
            if (ind.feasible) best_d_penalized = std::min(best_d_penalized, ind.d_penalized_mean);
        }
    }

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

        // Best-of-run weight snapshot: persist iff this genome's seed-winning weights beat
        // every feasible individual evaluated so far in this run. Checked/cleared BEFORE
        // caching so the weight tensors never linger in table/history for genomes that
        // were not (or are no longer) the best — only the current winner's file survives.
        if (!weights_path.empty() && ind.weights_snapshot && ind.feasible &&
            ind.d_penalized_mean < best_d_penalized)
        {
            save_state_dict_npz(*ind.weights_snapshot, weights_path);
            best_d_penalized = ind.d_penalized_mean;
        }
        ind.weights_snapshot.reset();

        // Persist BEFORE inserting so a genome recorded in memory is always on disk too
        // (layer 1: at most the single in-flight training is ever lost to a crash).
        if (!persist_path.empty()) append_cache_entry(persist_path, ind);

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

// Binary tournament that never returns `exclude` — used to pick the SECOND parent so an
// individual can never mate with itself (no self-mating). Requires pop.size() >= 2. Each
// draw is redrawn until it differs from `exclude`, so the winner is guaranteed distinct;
// this is an invariant, not a fallback (no data is guessed).
int tournament_excluding(const std::vector<Individual>& pop, std::mt19937& rng, int k, int exclude)
{
    std::uniform_int_distribution<int> pick(0, static_cast<int>(pop.size()) - 1);
    auto draw_other = [&]
    {
        int c = pick(rng);
        while (c == exclude) c = pick(rng);
        return c;
    };
    int best = draw_other();
    for (int i = 1; i < k; ++i)
    {
        int c = draw_other();
        if (crowded_less(pop[c], pop[best])) best = c;
    }
    return best;
}

// Build a population member from a diploid genotype: train/score its EXPRESSED
// phenotype (deduped by the cache), then attach this member's own genotype so it can
// reproduce. The cache stores phenotype-level results; two genotypes that express the
// same architecture share one training.
Individual eval_member(DiploidGenome geno,
    int gen,
    EvalCache& cache,
    const thesis::ThesisDatasetView& view,
    const GaConfig& cfg,
    const EvalCallback& on_eval)
{
    const Individual& evaluated = cache.get(geno.expressed(), gen, view, cfg, on_eval);
    Individual m = evaluated;     // copy phenotype metrics + objectives + feasibility
    m.genotype = std::move(geno); // this member's diploid genotype (for meiosis)
    return m;
}

void rerank_population(std::vector<Individual>& pop)
{
    auto fronts = fast_non_dominated_sort(pop);
    for (const auto& f : fronts) assign_crowding_distance(pop, f);
}

// Resumes the parent population + RNG state from a generation checkpoint when one exists,
// otherwise evaluates a fresh initial population. Returns the starting parents and the
// first generation the caller's generational loop should (re)compute.
std::pair<std::vector<Individual>, int> initialize_or_resume_population(EvalCache& cache,
    const thesis::ThesisDatasetView& view,
    const GaConfig& cfg,
    const EvalCallback& on_eval,
    std::mt19937& rng,
    bool ckpt,
    bool is_snn,
    nn::progress::ProgressManager& pm)
{
    const int N = cfg.ga.population_size;
    std::vector<Individual> parents;
    parents.reserve(static_cast<size_t>(N));
    int start_gen = 1; // first generation to (re)compute

    if (ckpt && state_checkpoint_exists(cfg.results_dir, cfg.run_tag))
    {
        // ── Resume ──────────────────────────────────────────────────────────
        GenerationCheckpoint ck = load_generation_checkpoint(cfg.results_dir, cfg.run_tag);
        rng_from_string(rng, ck.rng_state); // exact engine state at end of ck.generation
        parents = std::move(ck.parents);
        rerank_population(parents); // rank/crowding were not serialized; recompute for tournaments
        start_gen = ck.generation + 1;
        pm.log("[PGA] Resuming " + cfg.run_tag + " from checkpoint: generation " +
               std::to_string(ck.generation) + "/" + std::to_string(cfg.ga.generations) +
               " complete, " + std::to_string(cache.history.size()) +
               " genomes already in cache. Continuing at generation " + std::to_string(start_gen) +
               ".");
    }
    else
    {
        // ── Fresh start: initial population ─────────────────────────────────
        pm.log("[PGA] Generation 0: initializing + evaluating " + std::to_string(N) +
               " diploid individuals (" + cfg.base.feature_extraction.autoencoder.model + " / " +
               cfg.base.dataset.modality + ")");
        for (int i = 0; i < N; ++i)
            parents.push_back(eval_member(
                random_diploid(rng, cfg.ga.bounds, is_snn), 0, cache, view, cfg, on_eval));
        rerank_population(parents);
        if (ckpt) save_generation_checkpoint(cfg.results_dir, cfg.run_tag, 0, rng, parents);
    }

    return {std::move(parents), start_gen};
}

// Produces N offspring via binary-tournament parent selection + meiosis + gamete fusion,
// evaluating each child through the cache. There is no clone path — reproduction is always
// the fusion of two gametes from two DISTINCT parents (no self-mating).
std::vector<Individual> generate_offspring(const std::vector<Individual>& parents,
    EvalCache& cache,
    const thesis::ThesisDatasetView& view,
    const GaConfig& cfg,
    const EvalCallback& on_eval,
    std::mt19937& rng,
    bool is_snn,
    int gen)
{
    const int N = cfg.ga.population_size;
    std::vector<Individual> offspring;
    offspring.reserve(static_cast<size_t>(N));
    while (static_cast<int>(offspring.size()) < N)
    {
        // Two DISTINCT parents (no self-mating): the second tournament excludes the
        // first winner's index.
        const int a = tournament(parents, rng, cfg.ga.tournament_k);
        const int b = tournament_excluding(parents, rng, cfg.ga.tournament_k, a);

        // Sexual reproduction: each parent forms a gamete by meiosis; the gametes
        // fuse into the diploid child.
        const Gamete g1 = meiosis(parents[a].genotype,
            rng,
            cfg.ga.bounds,
            cfg.ga.crossover_prob,
            cfg.ga.mutation_prob,
            is_snn);
        const Gamete g2 = meiosis(parents[b].genotype,
            rng,
            cfg.ga.bounds,
            cfg.ga.crossover_prob,
            cfg.ga.mutation_prob,
            is_snn);
        offspring.push_back(eval_member(fuse(g1, g2), gen, cache, view, cfg, on_eval));
    }
    return offspring;
}

// μ+λ selection over `combined` (parents+offspring), split into an elite quota (best
// n_winners by rank then crowding — standard elitism) and a diversity reserve of the
// worst-ranked non-winners (the anti-local-optimum hedge; n_losers=0 → textbook NSGA-II).
std::vector<Individual> select_next_generation(const std::vector<Individual>& combined,
    const std::vector<std::vector<int>>& cfronts,
    int n_winners,
    int N)
{
    std::vector<char> taken(combined.size(), 0);
    std::vector<Individual> next;
    next.reserve(static_cast<size_t>(N));

    // (1) Winners: the best n_winners by (rank, then crowding) — standard elitism.
    for (const auto& front : cfronts)
    {
        if (static_cast<int>(next.size()) >= n_winners) break;
        if (static_cast<int>(next.size() + front.size()) <= n_winners)
        {
            for (int idx : front)
            {
                next.push_back(combined[idx]);
                taken[idx] = 1;
            }
        }
        else
        {
            std::vector<int> ordered(front);
            std::sort(ordered.begin(),
                ordered.end(),
                [&](int x, int y) { return combined[x].crowding > combined[y].crowding; });
            for (int idx : ordered)
            {
                if (static_cast<int>(next.size()) >= n_winners) break;
                next.push_back(combined[idx]);
                taken[idx] = 1;
            }
            break;
        }
    }

    // (2) Losers: fill the remaining n_losers slots from the NON-winners, worst
    // (highest) rank first, ties broken by larger crowding so the kept losers are the
    // most distinct available. Deliberately preserving losing material is the
    // anti-local-optimum hedge (n_losers=0 → textbook NSGA-II).
    if (static_cast<int>(next.size()) < N)
    {
        std::vector<int> remainder;
        for (int i = 0; i < static_cast<int>(combined.size()); ++i)
            if (!taken[i]) remainder.push_back(i);
        std::sort(remainder.begin(),
            remainder.end(),
            [&](int x, int y)
            {
                if (combined[x].rank != combined[y].rank)
                    return combined[x].rank > combined[y].rank;     // worst rank first
                return combined[x].crowding > combined[y].crowding; // most distinct first
            });
        for (int idx : remainder)
        {
            if (static_cast<int>(next.size()) >= N) break;
            next.push_back(combined[idx]);
        }
    }
    return next;
}

void log_generation_progress(nn::progress::ProgressManager& pm,
    int gen,
    const GaConfig& cfg,
    const std::vector<Individual>& parents,
    const EvalCache& cache)
{
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
        << cfg.ga.population_size << " feasible, best feasible d_penalized="
        << (n_feasible ? best_dpen : std::numeric_limits<double>::quiet_NaN()) << ", "
        << cache.table.size() << " distinct genomes evaluated";
    pm.log(oss.str());
}

// Assembles the final GaResult from the last generation's parents: the deterministic
// (final_population, history) records, plus the feasible rank-0 Pareto front, sorted by
// (inference cost, then d_penalized) for reproducible downstream reporting.
GaResult assemble_ga_result(
    const GaConfig& cfg, std::vector<Individual> parents, const EvalCache& cache)
{
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

} // namespace

GaResult run_nsga2(
    const thesis::ThesisDatasetView& view, const GaConfig& cfg, const EvalCallback& on_eval)
{
    auto& pm = nn::progress::ProgressManager::instance();
    std::mt19937 rng(cfg.ga.seed);
    const bool is_snn = cfg.is_snn_population();
    const int N = cfg.ga.population_size;

    EvalCache cache;

    const int n_losers = cfg.ga.n_losers; // diversity reserve (worst-ranked survivors)
    const int n_winners = N - n_losers;   // μ+λ elite slots (validate: n_winners >= 1)

    // ── Checkpointing (layer 1: live cache; layer 2: per-generation state) ─────
    const bool ckpt = cfg.checkpoint.enabled;
    // Best-of-run weight snapshot: unconditional (not gated by checkpoint.enabled — it is
    // a result artifact, not a resume artifact, and is never deleted on completion).
    cache.weights_path = cfg.results_dir + "/models/pga_" + cfg.run_tag + "_best.npz";
    if (ckpt)
    {
        thesis::ensure_dir(cfg.results_dir); // must exist before the first cache append
        cache.persist_path = checkpoint_cache_path(cfg.results_dir, cfg.run_tag);
        cache.preload(load_cache_entries(cache.persist_path)); // warm cache — no retraining
    }

    auto [parents, start_gen] =
        initialize_or_resume_population(cache, view, cfg, on_eval, rng, ckpt, is_snn, pm);

    // ── Generational loop ────────────────────────────────────────────────────
    for (int gen = start_gen; gen <= cfg.ga.generations; ++gen)
    {
        std::vector<Individual> offspring =
            generate_offspring(parents, cache, view, cfg, on_eval, rng, is_snn, gen);

        // μ+λ selection over parents+offspring, split into an elite quota and a loser
        // reserve.
        std::vector<Individual> combined;
        combined.reserve(parents.size() + offspring.size());
        combined.insert(combined.end(), parents.begin(), parents.end());
        combined.insert(combined.end(), offspring.begin(), offspring.end());

        auto cfronts = fast_non_dominated_sort(combined);
        for (const auto& front : cfronts) assign_crowding_distance(combined, front);

        parents = select_next_generation(combined, cfronts, n_winners, N);
        // Refresh rank/crowding for the surviving population (used by next tournament).
        rerank_population(parents);

        log_generation_progress(pm, gen, cfg, parents, cache);

        // Layer 2: persist the surviving population + RNG at the end of this generation.
        // The cache (layer 1) is already on disk per-individual; together they bound a
        // crash's cost to one in-flight training.
        if (ckpt && (gen % cfg.checkpoint.every_generations == 0 || gen == cfg.ga.generations))
            save_generation_checkpoint(cfg.results_dir, cfg.run_tag, gen, rng, parents);
    }

    // ── Assemble result ──────────────────────────────────────────────────────
    GaResult result = assemble_ga_result(cfg, parents, cache);

    // Success: the run is complete, so the resume artifacts are no longer needed. The
    // results (CSV + Pareto JSON) written by the caller are what persist.
    if (ckpt) remove_checkpoint_artifacts(cfg.results_dir, cfg.run_tag);

    return result;
}

} // namespace pga
