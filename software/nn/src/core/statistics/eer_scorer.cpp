#include "statistics/eer_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "statistics/confusion_matrix.hpp"

namespace statistics
{

namespace
{

// Cosine similarity between two equal-length vectors; returns 0 for zero vectors.
float cosine_sim(const std::vector<float>& a, const std::vector<float>& b)
{
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    const float denom = std::sqrt(na * nb);
    return (denom > 0.0f) ? (dot / denom) : 0.0f;
}

// L2-normalise a vector in-place.  No-op for zero vectors.
void l2_normalize(std::vector<float>& v)
{
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 0.0f)
        for (float& x : v) x /= norm;
}

// Mean of a set of vectors.  All vectors must be the same length.
std::vector<float> mean_vec(const std::vector<std::vector<float>>& vecs)
{
    if (vecs.empty()) return {};
    const std::size_t D = vecs[0].size();
    std::vector<float> m(D, 0.0f);
    for (const auto& v : vecs)
        for (std::size_t i = 0; i < D; ++i) m[i] += v[i];
    const float n = static_cast<float>(vecs.size());
    for (float& x : m) x /= n;
    return m;
}

// Compute EER from a list of (score, is_genuine) pairs via threshold sweep.
// Trials are sorted descending by score; FAR and FRR are swept until they cross.
// Returns NaN if no genuine or no impostor trials are present.
double eer_from_trials(std::vector<std::pair<float, bool>> trials)
{
    if (trials.empty()) return std::numeric_limits<double>::quiet_NaN();

    std::sort(trials.begin(), trials.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    int n_gen = 0;
    int n_imp = 0;
    for (const auto& [s, g] : trials)
    {
        if (g) ++n_gen;
        else   ++n_imp;
    }
    if (n_gen == 0 || n_imp == 0) return std::numeric_limits<double>::quiet_NaN();

    // Start: threshold above all scores → accept all → FA=n_imp, FR=0
    int fa = n_imp;
    int fr = 0;
    double prev_far = 1.0;
    double prev_frr = 0.0;

    std::size_t i = 0;
    while (i < trials.size())
    {
        // Process all tied scores together.
        const float thr = trials[i].first;
        while (i < trials.size() && trials[i].first == thr)
        {
            if (trials[i].second) ++fr;
            else                  --fa;
            ++i;
        }

        const double far = static_cast<double>(fa) / static_cast<double>(n_imp);
        const double frr = static_cast<double>(fr) / static_cast<double>(n_gen);

        if (far <= frr)
        {
            // Linear interpolation between previous and current (far, frr).
            const double d_far = prev_far - far;
            const double d_frr = frr - prev_frr;
            const double denom = d_far + d_frr;
            if (denom > 0.0)
                return prev_frr + d_frr * d_far / denom;
            return (far + frr) / 2.0;
        }
        prev_far = far;
        prev_frr = frr;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// AUC via Wilcoxon-Mann-Whitney statistic: P(genuine_score > impostor_score).
// Ties count 0.5.  O(G log I) via sorted impostors + binary search.
double mann_whitney_auc(const std::vector<std::pair<float, bool>>& trials)
{
    std::vector<float> genuine;
    std::vector<float> impostor;
    for (const auto& [s, g] : trials)
        (g ? genuine : impostor).push_back(s);

    if (genuine.empty() || impostor.empty())
        return std::numeric_limits<double>::quiet_NaN();

    std::sort(impostor.begin(), impostor.end());

    double wins = 0.0;
    for (float gs : genuine)
    {
        // impostors strictly below gs
        auto lt = static_cast<double>(
            std::lower_bound(impostor.begin(), impostor.end(), gs) - impostor.begin());
        // impostors equal to gs
        auto eq = static_cast<double>(
            std::upper_bound(impostor.begin(), impostor.end(), gs) - impostor.begin()) - lt;
        wins += lt + 0.5 * eq;
    }
    return wins / (static_cast<double>(genuine.size()) * static_cast<double>(impostor.size()));
}

// Build genuine/impostor cosine-similarity trial list for GenuineImpostorEERScorer.
// Returns empty vector when too few samples or speakers.
std::vector<std::pair<float, bool>> build_gi_trials(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    std::size_t n_enroll)
{
    if (embeddings.empty()) return {};

    // Group sample indices by speaker (stable insertion order).
    std::vector<int> speaker_order;
    std::unordered_map<int, std::vector<std::size_t>> spk_to_indices;
    for (std::size_t i = 0; i < labels.size(); ++i)
    {
        if (spk_to_indices.find(labels[i]) == spk_to_indices.end())
            speaker_order.push_back(labels[i]);
        spk_to_indices[labels[i]].push_back(i);
    }

    // Build enrollment templates.
    std::unordered_map<int, std::vector<float>> templates;
    std::unordered_map<int, std::vector<std::size_t>> probe_indices;

    for (int spk : speaker_order)
    {
        const auto& idxs = spk_to_indices.at(spk);
        if (idxs.size() <= n_enroll) continue;

        std::vector<std::vector<float>> enroll_vecs;
        enroll_vecs.reserve(n_enroll);
        for (std::size_t k = 0; k < n_enroll; ++k)
            enroll_vecs.push_back(embeddings[idxs[k]]);

        auto tmpl = mean_vec(enroll_vecs);
        l2_normalize(tmpl);
        templates[spk] = std::move(tmpl);

        probe_indices[spk] = std::vector<std::size_t>(
            idxs.begin() + static_cast<ptrdiff_t>(n_enroll), idxs.end());
    }

    if (templates.size() < 2U) return {};

    // Score all probes against all templates.
    std::vector<std::pair<float, bool>> trials;
    for (int spk : speaker_order)
    {
        if (probe_indices.find(spk) == probe_indices.end()) continue;
        for (std::size_t probe_idx : probe_indices.at(spk))
        {
            auto probe = embeddings[probe_idx];
            l2_normalize(probe);
            for (int tmpl_spk : speaker_order)
            {
                if (templates.find(tmpl_spk) == templates.end()) continue;
                trials.emplace_back(cosine_sim(probe, templates.at(tmpl_spk)),
                    tmpl_spk == spk);
            }
        }
    }
    return trials;
}

} // namespace

// ── ClassificationEERScorer ───────────────────────────────────────────────────

auto ClassificationEERScorer::compute_eer(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int n_classes) const -> double
{
    if (embeddings.empty()) return std::numeric_limits<double>::quiet_NaN();

    std::vector<ConfusionMatrix> cms(static_cast<std::size_t>(n_classes));

    for (std::size_t i = 0; i < embeddings.size(); ++i)
    {
        // argmax over embedding row
        int pred   = 0;
        float best = embeddings[i][0];
        for (int j = 1; j < n_classes; ++j)
        {
            if (embeddings[i][static_cast<std::size_t>(j)] > best)
            {
                best = embeddings[i][static_cast<std::size_t>(j)];
                pred = j;
            }
        }

        for (int c = 0; c < n_classes; ++c)
        {
            const bool is_c   = (labels[i] == c);
            const bool pred_c = (pred == c);
            auto& cm = cms[static_cast<std::size_t>(c)];
            if ( is_c &&  pred_c) ++cm.truePositive;
            if (!is_c &&  pred_c) ++cm.falsePositive;
            if ( is_c && !pred_c) ++cm.falseNegative;
            if (!is_c && !pred_c) ++cm.trueNegative;
        }
    }

    double eer = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> fprs;
    std::vector<double> fnrs;
    calculateEER(cms, eer, fprs, fnrs);
    return eer;
}

// ── GenuineImpostorEERScorer ──────────────────────────────────────────────────

GenuineImpostorEERScorer::GenuineImpostorEERScorer(std::size_t n_enroll_per_speaker)
    : n_enroll_(n_enroll_per_speaker)
{
}

auto GenuineImpostorEERScorer::compute_eer(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int /*n_classes*/) const -> double
{
    auto trials = build_gi_trials(embeddings, labels, n_enroll_);
    if (trials.empty()) return std::numeric_limits<double>::quiet_NaN();
    return eer_from_trials(std::move(trials));
}

auto GenuineImpostorEERScorer::compute_auc(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int /*n_classes*/) const -> double
{
    const auto trials = build_gi_trials(embeddings, labels, n_enroll_);
    if (trials.empty()) return std::numeric_limits<double>::quiet_NaN();
    return mann_whitney_auc(trials);
}

} // namespace statistics
