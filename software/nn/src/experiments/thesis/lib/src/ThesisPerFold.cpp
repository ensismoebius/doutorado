/**
 * @file src/experiments/thesis/lib/src/ThesisPerFold.cpp
 * @brief The per-fold protocol (ThesisPerFold.hpp, .wiki/Experiments/Thesis-Per-Fold-Protocol.md).
 */
#include "ThesisPerFold.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "IndexIntersection.hpp"
#include "ThesisClassifiers.hpp"
#include "ThesisParaconsistent.hpp"
#include "progress/ProgressManager.hpp"
#include "statistics/GroupKFoldPolicy.hpp"

namespace thesis
{

namespace
{

constexpr std::uint32_t kInnerSeedMix = 0xDEADBEEFU; // the nested CV's inner-fold seed rule

auto sorted_text_split(const std::vector<ThesisSample>& samples, const ThesisConfig& cfg)
    -> TextSplit
{
    TextSplit ts = make_text_split(samples, cfg.classifier.text_mode, cfg.experiment.seed);
    std::sort(ts.train_indices.begin(), ts.train_indices.end());
    std::sort(ts.test_indices.begin(), ts.test_indices.end());
    return ts;
}

auto speaker_groups(const std::vector<ThesisSample>& samples) -> std::vector<int>
{
    std::vector<int> groups;
    groups.reserve(samples.size());
    for (const auto& s : samples) groups.push_back(s.subject_id);
    return groups;
}

auto fmt_float(float v) -> std::string
{
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

auto point_name(const GridPoint& p) -> std::string
{
    return "width " + std::to_string(p.hidden_width) + ", blocks " +
           std::to_string(p.residual_blocks) + ", lr " + fmt_float(p.learning_rate) + ", wd " +
           fmt_float(p.weight_decay);
}

/// The one FeatureSet a candidate config extracts.
auto only_set(std::vector<FeatureSet> sets, std::size_t n_samples, std::size_t candidate)
    -> FeatureSet
{
    if (sets.size() != 1)
        throw std::logic_error("ThesisPerFold: extractor candidate " + std::to_string(candidate) +
                               " produced " + std::to_string(sets.size()) + " feature sets, not 1");
    if (sets[0].vectors.size() != n_samples)
        throw std::logic_error("ThesisPerFold: extractor candidate " + std::to_string(candidate) +
                               " produced " + std::to_string(sets[0].vectors.size()) +
                               " vectors for " + std::to_string(n_samples) + " samples");
    return std::move(sets[0]);
}

auto is_handcrafted(const ThesisConfig& candidate) -> bool
{
    return candidate.feature_extraction.strategy == "handcrafted";
}

/// The features of candidate i for `modality`, its fitted parts (PCA, autoencoder) fitted on
/// `fit_rows` only. Handcrafted features fit nothing and come from the per-process cache.
auto candidate_features(const ThesisDatasetView& view,
    const ThesisConfig& cfg,
    std::size_t i,
    const std::string& modality,
    const std::vector<std::size_t>& fit_rows,
    std::map<std::pair<std::size_t, std::string>, FeatureSet>& cache) -> FeatureSet
{
    const ThesisConfig cand = cfg.candidate_config(i, modality);
    const std::size_t n = view.samples.size();
    if (!is_handcrafted(cand))
        return only_set(extract_features_fitted_on(view, cand, fit_rows, nullptr), n, i);
    const auto key = std::make_pair(i, modality);
    auto it = cache.find(key);
    if (it == cache.end())
        it = cache.emplace(key, only_set(extract_features(view, cand, nullptr), n, i)).first;
    return it->second;
}

/// The degraded twin of the fold's winner features. A fitted extractor (PCA, autoencoder) must
/// be the SAME model the clean features came from: it is refitted on a view whose training rows
/// keep the clean voice (the rows it is fitted on) and whose other rows carry the degraded one.
/// The fit reads only the training rows and seeds every frame by the sample's own index, so
/// the refit is the clean fit; the training rows of both extractions are compared bit for bit
/// to prove it, and a difference stops the run. Only test rows are read from the result.
auto degraded_features(const ThesisDatasetView& clean_view,
    const ThesisDatasetView& degraded_view,
    const ThesisConfig& cfg,
    std::size_t winner,
    const std::string& modality,
    const std::vector<std::size_t>& fit_rows,
    const FeatureSet& clean,
    CandidateFeatureCache& cache) -> FeatureSet
{
    const ThesisConfig cand = cfg.candidate_config(winner, modality);
    if (is_handcrafted(cand))
        return candidate_features(degraded_view, cfg, winner, modality, fit_rows, cache.degraded);

    ThesisDatasetView hybrid = degraded_view;
    for (std::size_t r : fit_rows) hybrid.samples.at(r).audio = clean_view.samples.at(r).audio;
    FeatureSet out = only_set(extract_features_fitted_on(hybrid, cand, fit_rows, nullptr),
        clean_view.samples.size(),
        winner);
    for (std::size_t r : fit_rows)
        if (out.vectors[r] != clean.vectors[r])
            throw std::runtime_error("ThesisPerFold: the degraded extraction of \"" + clean.label +
                                     "\" did not refit the clean model (training row " +
                                     std::to_string(r) + " differs)");
    return out;
}

/// Mean inner EER of every grid point, the jobs (point, inner fold) run in parallel; each job's
/// result goes to its own slot, so the outcome does not depend on the thread schedule.
void score_grid(const ThesisDatasetView& view,
    const FeatureSet& features,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& scorer,
    const std::vector<InnerFoldRows>& inner,
    std::size_t outer_idx,
    std::vector<GridPoint>& grid)
{
    const auto n_inner = inner.size();
    std::vector<ThesisConfig> configs;
    configs.reserve(grid.size());
    for (auto& p : grid)
    {
        configs.push_back(grid_config(cfg, p));
        p.param_count = classifier_param_count(view, features.vectors, configs.back(), scorer);
        p.inner_eer.assign(n_inner, std::numeric_limits<double>::quiet_NaN());
    }

    const auto n_jobs = static_cast<long>(grid.size() * n_inner);
    std::exception_ptr first_error;
    long first_error_job = -1;
#pragma omp parallel for schedule(dynamic, 1)
    for (long job = 0; job < n_jobs; ++job)
    {
        const auto g = static_cast<std::size_t>(job) / n_inner;
        const auto j = static_cast<std::size_t>(job) % n_inner;
        try
        {
            SplitRun split;
            split.fold = static_cast<int>(j);
            split.total_folds = static_cast<int>(n_inner);
            split.save_model = false;
            const FoldResult fr = run_split(view,
                features.vectors,
                features.label,
                configs[g],
                scorer,
                inner[j].train,
                inner[j].validation,
                split);
            grid[g].inner_eer[j] = fr.eer;
        }
        catch (...)
        {
#pragma omp critical(thesis_per_fold_grid_error)
            if (!first_error || job < first_error_job)
            {
                first_error = std::current_exception();
                first_error_job = job;
            }
        }
    }
    if (first_error)
    {
        const auto g = static_cast<std::size_t>(first_error_job) / n_inner;
        const auto j = static_cast<std::size_t>(first_error_job) % n_inner;
        try
        {
            std::rethrow_exception(first_error);
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("ThesisPerFold: outer fold " + std::to_string(outer_idx) +
                                     ", grid point " + point_name(grid[g]) + ", inner fold " +
                                     std::to_string(j) + ": " + e.what());
        }
    }
    for (auto& p : grid)
    {
        double sum = 0.0;
        for (double e : p.inner_eer)
        {
            if (!std::isfinite(e))
                throw std::logic_error(
                    "ThesisPerFold: grid point " + point_name(p) + " has a non-finite inner EER");
            sum += e;
        }
        p.mean_inner_eer = sum / static_cast<double>(n_inner);
    }
}

} // namespace

auto outer_fold_rows(const std::vector<ThesisSample>& samples, const ThesisConfig& cfg)
    -> std::vector<OuterFoldRows>
{
    const TextSplit ts = sorted_text_split(samples, cfg);
    const statistics::GroupKFoldPolicy policy(
        static_cast<std::size_t>(cfg.training.k_folds), true, cfg.experiment.seed);
    const auto splits = policy.make_splits(samples.size(), speaker_groups(samples));
    std::vector<OuterFoldRows> out;
    out.reserve(splits.size());
    for (std::size_t f = 0; f < splits.size(); ++f)
    {
        OuterFoldRows rows;
        rows.speaker_rows = splits[f].train_indices;
        std::sort(rows.speaker_rows.begin(), rows.speaker_rows.end());
        rows.train = intersect_indices(rows.speaker_rows, ts.train_indices);
        rows.test = intersect_indices(splits[f].test_indices, ts.test_indices);
        if (rows.train.empty() || rows.test.empty())
            throw std::runtime_error("ThesisPerFold: outer fold " + std::to_string(f) + " has " +
                                     std::to_string(rows.train.size()) + " train and " +
                                     std::to_string(rows.test.size()) +
                                     " test rows after the text split");
        out.push_back(std::move(rows));
    }
    return out;
}

auto inner_fold_rows(
    const std::vector<ThesisSample>& samples, const OuterFoldRows& outer, const ThesisConfig& cfg)
    -> std::vector<InnerFoldRows>
{
    const TextSplit ts = sorted_text_split(samples, cfg);
    std::vector<int> groups;
    groups.reserve(outer.speaker_rows.size());
    for (std::size_t r : outer.speaker_rows) groups.push_back(samples.at(r).subject_id);

    const statistics::GroupKFoldPolicy policy(static_cast<std::size_t>(cfg.selection.inner_k.get()),
        true,
        cfg.experiment.seed ^ kInnerSeedMix);
    const auto splits = policy.make_splits(outer.speaker_rows.size(), groups);
    const auto to_rows = [&](const std::vector<std::size_t>& local)
    {
        std::vector<std::size_t> rows;
        rows.reserve(local.size());
        for (std::size_t k : local) rows.push_back(outer.speaker_rows.at(k));
        return rows;
    };
    std::vector<InnerFoldRows> out;
    out.reserve(splits.size());
    for (std::size_t j = 0; j < splits.size(); ++j)
    {
        InnerFoldRows rows;
        rows.train = intersect_indices(to_rows(splits[j].train_indices), ts.train_indices);
        rows.validation = intersect_indices(to_rows(splits[j].test_indices), ts.test_indices);
        if (rows.train.empty() || rows.validation.empty())
            throw std::runtime_error("ThesisPerFold: inner fold " + std::to_string(j) + " has " +
                                     std::to_string(rows.train.size()) + " train and " +
                                     std::to_string(rows.validation.size()) +
                                     " validation rows after the text split");
        out.push_back(std::move(rows));
    }
    return out;
}

auto rank_extractors(const std::vector<ThesisSample>& samples,
    const std::vector<std::size_t>& rows,
    const std::vector<FeatureSet>& candidates) -> std::vector<ExtractorRank>
{
    if (rows.empty()) throw std::invalid_argument("ThesisPerFold: no rows to rank extractors on");
    if (candidates.empty()) throw std::invalid_argument("ThesisPerFold: no extractor candidates");
    std::vector<ThesisSample> sub_samples;
    sub_samples.reserve(rows.size());
    for (std::size_t i : rows)
    {
        ThesisSample s; // labels only: the score never reads the signals
        s.subject_id = samples.at(i).subject_id;
        s.stimulus = samples.at(i).stimulus;
        s.text_phrase = samples.at(i).text_phrase;
        sub_samples.push_back(std::move(s));
    }
    std::vector<ExtractorRank> ranks;
    ranks.reserve(candidates.size());
    for (const auto& fs : candidates)
    {
        FeatureSet sub{fs.label, {}};
        sub.vectors.reserve(rows.size());
        for (std::size_t i : rows) sub.vectors.push_back(fs.vectors.at(i));
        const double d = score_feature_set(sub_samples, sub).d_penalized;
        if (!std::isfinite(d))
            throw std::runtime_error(
                "ThesisPerFold: extractor \"" + fs.label + "\" has a non-finite D_pen");
        ranks.push_back({fs.label, d});
    }
    return ranks;
}

auto extractor_winner(const std::vector<ExtractorRank>& ranks) -> std::size_t
{
    if (ranks.empty()) throw std::invalid_argument("ThesisPerFold: no extractor to choose from");
    std::size_t best = 0;
    for (std::size_t i = 1; i < ranks.size(); ++i)
        if (ranks[i].d_penalized < ranks[best].d_penalized) best = i;
    return best;
}

auto grid_points(const ThesisConfig::Selection& selection) -> std::vector<GridPoint>
{
    std::vector<GridPoint> grid;
    for (int w : selection.hidden_widths.get())
        for (int b : selection.residual_blocks.get())
            for (float lr : selection.learning_rates.get())
                for (float wd : selection.weight_decays.get())
                {
                    GridPoint p;
                    p.hidden_width = w;
                    p.residual_blocks = b;
                    p.learning_rate = lr;
                    p.weight_decay = wd;
                    grid.push_back(p);
                }
    if (grid.empty()) throw std::invalid_argument("ThesisPerFold: the grid is empty");
    return grid;
}

auto grid_config(const ThesisConfig& cfg, const GridPoint& point) -> ThesisConfig
{
    ThesisConfig c = cfg;
    c.classifier.layer_spec =
        std::vector<std::string>{"linear:" + std::to_string(point.hidden_width) + ":relu",
            "residual:" + std::to_string(point.residual_blocks + 1),
            "linear:N_speakers:identity"};
    c.training.learning_rate = point.learning_rate;
    c.training.weight_decay = point.weight_decay;
    return c;
}

auto grid_winner(const std::vector<GridPoint>& grid) -> std::size_t
{
    if (grid.empty()) throw std::invalid_argument("ThesisPerFold: no grid point to choose from");
    std::size_t best = 0;
    for (std::size_t g = 1; g < grid.size(); ++g)
    {
        const auto& p = grid[g];
        const auto& b = grid[best];
        if (p.mean_inner_eer < b.mean_inner_eer ||
            (p.mean_inner_eer == b.mean_inner_eer && p.param_count < b.param_count))
            best = g;
    }
    return best;
}

auto per_fold_label(const ThesisConfig& cfg) -> std::string
{
    const std::string& modality = cfg.dataset.modality;
    if (modality == "fused") return "per-fold-fused-" + cfg.dataset.fusion_mode.get();
    return "per-fold-" + modality;
}

auto run_per_fold(const ThesisDatasetView& view,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& scorer,
    const PerFoldDegradation* degradation,
    CandidateFeatureCache& cache,
    uint32_t global_bar_id) -> ClassificationResult
{
    if (!cfg.selection.enabled)
        throw std::logic_error("ThesisPerFold: the profile does not enable selection");
    if (degradation != nullptr &&
        (degradation->view == nullptr || degradation->view->samples.size() != view.samples.size()))
        throw std::invalid_argument("ThesisPerFold: the degraded view is not the clean one's twin");

    auto& pm = nn::progress::ProgressManager::instance();
    const std::string& modality = cfg.dataset.modality;
    const std::string& rank_modality = cfg.selection.rank_modality;
    const bool network = cfg.classifier.type != "cosine";
    const std::size_t n_candidates = cfg.selection.extractor_candidates.size();
    const auto folds = outer_fold_rows(view.samples, cfg);
    const int total = static_cast<int>(folds.size());

    ClassificationResult result;
    result.feature_set_label = per_fold_label(cfg);
    result.classifier_type = cfg.classifier.type;
    result.text_mode = cfg.classifier.text_mode;

    for (std::size_t f = 0; f < folds.size(); ++f)
    {
        const OuterFoldRows& rows = folds[f];
        const auto t0 = std::chrono::steady_clock::now();
        FoldSelection sel;

        // 1-2. Every candidate on the fold's training rows; the winner by D_pen on them.
        std::vector<FeatureSet> ranked;
        ranked.reserve(n_candidates);
        for (std::size_t i = 0; i < n_candidates; ++i)
            ranked.push_back(
                candidate_features(view, cfg, i, rank_modality, rows.train, cache.clean));
        sel.extractors = rank_extractors(view.samples, rows.train, ranked);
        sel.extractor_index = extractor_winner(sel.extractors);

        // 3. The winner's features of the profile's own signal.
        FeatureSet features =
            modality == rank_modality
                ? std::move(ranked[sel.extractor_index])
                : candidate_features(
                      view, cfg, sel.extractor_index, modality, rows.train, cache.clean);
        ranked.clear();

        // 4. The grid on the inner folds (the cosine control has nothing to tune).
        ThesisConfig fold_cfg = cfg;
        if (network)
        {
            sel.grid = grid_points(cfg.selection);
            score_grid(
                view, features, cfg, scorer, inner_fold_rows(view.samples, rows, cfg), f, sel.grid);
            sel.grid_index = grid_winner(sel.grid);
            fold_cfg = grid_config(cfg, sel.grid[sel.grid_index]);
            sel.param_count = sel.grid[sel.grid_index].param_count;
        }

        // 5. Refit on every training row, score the test rows once.
        std::optional<FeatureSet> degraded;
        std::optional<DegradedTestFeatures> degraded_test;
        if (degradation != nullptr)
        {
            degraded = degraded_features(view,
                *degradation->view,
                cfg,
                sel.extractor_index,
                modality,
                rows.train,
                features,
                cache);
            degraded_test = DegradedTestFeatures{
                &degraded->vectors, degradation->enrol_clean, degradation->enrol_per_speaker};
        }
        SplitRun split;
        split.fold = static_cast<int>(f);
        split.total_folds = total;
        split.save_model = network;
        split.degraded = degraded_test ? &*degraded_test : nullptr;
        split.global_bar_id = global_bar_id;
        FoldResult fr = run_split(view,
            features.vectors,
            result.feature_set_label,
            fold_cfg,
            scorer,
            rows.train,
            rows.test,
            split);
        // train_ms stays the refit's (comparable with the fixed-configuration run); the
        // choosing — extraction, ranking, search, refit and test — is timed as a whole.
        sel.fold_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count();
        fr.selection = std::move(sel);

        {
            const auto& s = *fr.selection;
            std::ostringstream oss;
            oss << "[Thesis] per-fold " << f << ": extractor " << s.extractor_index << " ("
                << s.extractors[s.extractor_index].label << ", D_pen "
                << s.extractors[s.extractor_index].d_penalized << ")";
            if (network)
                oss << ", " << point_name(s.grid[s.grid_index]) << " (inner EER "
                    << s.grid[s.grid_index].mean_inner_eer << ")";
            oss << ", test EER " << fr.eer;
            pm.log(oss.str());
        }
        result.outer_folds.push_back(std::move(fr));
        if (global_bar_id != 0) pm.update_bar(global_bar_id, static_cast<float>(f + 1));
    }

    compute_aggregate_stats(result);
    result.param_count = std::nullopt; // per fold: FoldResult::selection->param_count
    return result;
}

} // namespace thesis
