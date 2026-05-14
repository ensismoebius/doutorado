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
    if (embeddings.empty()) return std::numeric_limits<double>::quiet_NaN();

    // Group sample indices by speaker label (stable insertion order).
    std::vector<int> speaker_order;
    std::unordered_map<int, std::vector<std::size_t>> spk_to_indices;
    for (std::size_t i = 0; i < labels.size(); ++i)
    {
        if (spk_to_indices.find(labels[i]) == spk_to_indices.end())
            speaker_order.push_back(labels[i]);
        spk_to_indices[labels[i]].push_back(i);
    }

    // Build enrollment template per speaker.
    std::unordered_map<int, std::vector<float>> templates;
    std::unordered_map<int, std::vector<std::size_t>> probe_indices;

    for (int spk : speaker_order)
    {
        const auto& idxs = spk_to_indices.at(spk);
        if (idxs.size() <= n_enroll_) continue; // not enough probes — skip

        // Enrollment: first n_enroll_ samples → mean → L2-normalise.
        std::vector<std::vector<float>> enroll_vecs;
        enroll_vecs.reserve(n_enroll_);
        for (std::size_t k = 0; k < n_enroll_; ++k)
            enroll_vecs.push_back(embeddings[idxs[k]]);

        auto tmpl = mean_vec(enroll_vecs);
        l2_normalize(tmpl);
        templates[spk] = std::move(tmpl);

        // Probes: remaining samples.
        std::vector<std::size_t> probes(idxs.begin() + static_cast<ptrdiff_t>(n_enroll_),
            idxs.end());
        probe_indices[spk] = std::move(probes);
    }

    if (templates.size() < 2U)
        return std::numeric_limits<double>::quiet_NaN(); // need >= 2 speakers for impostors

    // Build trial list: (cosine_score, is_genuine).
    std::vector<std::pair<float, bool>> trials;

    for (int spk : speaker_order)
    {
        if (probe_indices.find(spk) == probe_indices.end()) continue;

        for (std::size_t probe_idx : probe_indices.at(spk))
        {
            // L2-normalise probe embedding.
            auto probe = embeddings[probe_idx];
            l2_normalize(probe);

            for (int tmpl_spk : speaker_order)
            {
                if (templates.find(tmpl_spk) == templates.end()) continue;
                const float score = cosine_sim(probe, templates.at(tmpl_spk));
                const bool is_genuine = (tmpl_spk == spk);
                trials.emplace_back(score, is_genuine);
            }
        }
    }

    return eer_from_trials(std::move(trials));
}

} // namespace statistics
