#include "statistics/eer_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "statistics/ClassificationEERScorer.hpp"
#include "statistics/GenuineImpostorEERScorer.hpp"

namespace statistics
{

namespace
{

// Every scoring precondition below throws instead of substituting a value. Until
// 2026-11-12 a zero vector scored cosine 0 against everything, a speaker with too few
// samples was left out of the trials, and an unscorable set came back as NaN — each of
// them a number, or an absence, that looked like a result.
[[noreturn]] void refuse(const std::string& why)
{
    throw std::invalid_argument("EER scorer: " + why);
}

// Cosine similarity between two equal-length, nonzero, finite vectors.
float cosine_sim(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size())
        refuse("cosine of vectors of length " + std::to_string(a.size()) + " and " +
               std::to_string(b.size()));
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    const float denom = std::sqrt(na * nb);
    if (!(denom > 0.0f) || !std::isfinite(dot / denom))
        refuse("cosine is undefined for a zero or non-finite embedding");
    return dot / denom;
}

// L2-normalise a vector in-place; a zero or non-finite vector has no direction.
void l2_normalize(std::vector<float>& v)
{
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (!(norm > 0.0f) || !std::isfinite(norm))
        refuse("cannot L2-normalise a zero or non-finite embedding");
    for (float& x : v) x /= norm;
}

// Mean of a non-empty set of equal-length vectors.
std::vector<float> mean_vec(const std::vector<std::vector<float>>& vecs)
{
    if (vecs.empty()) refuse("mean of no vectors");
    const std::size_t D = vecs[0].size();
    std::vector<float> m(D, 0.0f);
    for (const auto& v : vecs)
    {
        if (v.size() != D) refuse("enrolment vectors of different lengths");
        for (std::size_t i = 0; i < D; ++i) m[i] += v[i];
    }
    const float n = static_cast<float>(vecs.size());
    for (float& x : m) x /= n;
    return m;
}

// Compute EER from a list of (score, is_genuine) pairs via threshold sweep.
// Trials are sorted descending by score; FAR and FRR are swept until they cross.
// build_gi_trials guarantees both genuine and impostor trials.
double eer_from_trials(std::vector<std::pair<float, bool>> trials)
{
    std::sort(trials.begin(),
        trials.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    int n_gen = 0;
    int n_imp = 0;
    for (const auto& [s, g] : trials)
    {
        if (g)
            ++n_gen;
        else
            ++n_imp;
    }
    if (n_gen == 0 || n_imp == 0)
        refuse(std::to_string(n_gen) + " genuine and " + std::to_string(n_imp) +
               " impostor trials; an EER needs both");

    // A trial is accepted when its score is at or above the threshold (higher score = more
    // alike). The threshold starts above every score, so nothing is accepted: FA = 0,
    // FR = n_gen. Lowering it past each tied group of scores (descending) accepts that group:
    // a genuine trial stops being a false reject, an impostor becomes a false accept.
    //
    // Regression note: until 2026-11-12 this sweep started at FA = n_imp, FR = 0 and moved
    // the other way, i.e. it accepted LOW scores. A perfect system (every genuine above every
    // impostor) got EER 0.75 on four trials per side and a fully inverted one got 0, while
    // compute_auc (P(genuine > impostor)) was right. The unit tests used tied scores, the one
    // input on which both sweeps agree.
    int fa = 0;
    int fr = n_gen;
    double prev_far = 0.0;
    double prev_frr = 1.0;

    std::size_t i = 0;
    while (i < trials.size())
    {
        // Process all tied scores together.
        const float thr = trials[i].first;
        while (i < trials.size() && trials[i].first == thr)
        {
            if (trials[i].second)
                --fr;
            else
                ++fa;
            ++i;
        }

        const double far = static_cast<double>(fa) / static_cast<double>(n_imp);
        const double frr = static_cast<double>(fr) / static_cast<double>(n_gen);

        if (far >= frr)
        {
            // FAR rose and FRR fell along the segment from the previous operating point;
            // EER is where the two lines meet on it.
            const double d_far = far - prev_far;
            const double d_frr = prev_frr - frr;
            const double denom = d_far + d_frr;
            if (denom > 0.0) return prev_far + d_far * (prev_frr - prev_far) / denom;
            return (far + frr) / 2.0;
        }
        prev_far = far;
        prev_frr = frr;
    }
    // Unreachable: after the last group FAR = 1 >= FRR = 0.
    throw std::logic_error("EER scorer: the FAR/FRR sweep ended without crossing");
}

// AUC via Wilcoxon-Mann-Whitney statistic: P(genuine_score > impostor_score).
// Ties count 0.5.  O(G log I) via sorted impostors + binary search.
double mann_whitney_auc(const std::vector<std::pair<float, bool>>& trials)
{
    std::vector<float> genuine;
    std::vector<float> impostor;
    for (const auto& [s, g] : trials) (g ? genuine : impostor).push_back(s);

    if (genuine.empty() || impostor.empty())
        refuse(std::to_string(genuine.size()) + " genuine and " + std::to_string(impostor.size()) +
               " impostor trials; an AUC needs both");

    std::sort(impostor.begin(), impostor.end());

    double wins = 0.0;
    for (float gs : genuine)
    {
        // impostors strictly below gs
        auto lt = static_cast<double>(
            std::lower_bound(impostor.begin(), impostor.end(), gs) - impostor.begin());
        // impostors equal to gs
        auto eq = static_cast<double>(
                      std::upper_bound(impostor.begin(), impostor.end(), gs) - impostor.begin()) -
                  lt;
        wins += lt + 0.5 * eq;
    }
    return wins / (static_cast<double>(genuine.size()) * static_cast<double>(impostor.size()));
}

// Build genuine/impostor cosine-similarity trial list for GenuineImpostorEERScorer.
// Every speaker in the set must have more than n_enroll samples (enrolment + at least
// one probe), and there must be at least two speakers, or there is nothing to score.
std::vector<std::pair<float, bool>> build_gi_trials(
    const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    std::size_t n_enroll)
{
    if (n_enroll < 1) refuse("n_enroll must be at least 1");
    if (embeddings.empty()) refuse("no embeddings to score");
    if (embeddings.size() != labels.size())
        refuse(std::to_string(embeddings.size()) + " embeddings but " +
               std::to_string(labels.size()) + " labels");

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
        // This speaker used to be left out of the trials without a word: the EER was then
        // over fewer speakers than the evaluation set held.
        if (idxs.size() <= n_enroll)
            refuse("speaker " + std::to_string(spk) + " has " + std::to_string(idxs.size()) +
                   " sample(s), but " + std::to_string(n_enroll) +
                   " enrolment + at least 1 probe are needed");

        std::vector<std::vector<float>> enroll_vecs;
        enroll_vecs.reserve(n_enroll);
        for (std::size_t k = 0; k < n_enroll; ++k) enroll_vecs.push_back(embeddings[idxs[k]]);

        auto tmpl = mean_vec(enroll_vecs);
        l2_normalize(tmpl);
        templates[spk] = std::move(tmpl);

        probe_indices[spk] =
            std::vector<std::size_t>(idxs.begin() + static_cast<ptrdiff_t>(n_enroll), idxs.end());
    }

    if (templates.size() < 2U)
        refuse(std::to_string(templates.size()) +
               " speaker(s) in the set; impostor trials need at least 2");

    // Score all probes against all templates. Every speaker is enrolled (or the loop
    // above threw), so .at() cannot miss.
    std::vector<std::pair<float, bool>> trials;
    for (int spk : speaker_order)
    {
        for (std::size_t probe_idx : probe_indices.at(spk))
        {
            auto probe = embeddings[probe_idx];
            l2_normalize(probe);
            for (int tmpl_spk : speaker_order)
                trials.emplace_back(cosine_sim(probe, templates.at(tmpl_spk)), tmpl_spk == spk);
        }
    }
    return trials;
}

} // namespace

// ── ClassificationEERScorer ───────────────────────────────────────────────────

// Upgraded (audit m-4): use the genuine/impostor cosine-similarity method
// instead of the legacy closed-set confusion-matrix EER. One enrollment
// utterance per speaker, matching GenuineImpostorEERScorer's default.
auto ClassificationEERScorer::compute_eer(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int /*n_classes*/) const -> double
{
    auto trials = build_gi_trials(embeddings, labels, 1U);
    return eer_from_trials(std::move(trials));
}

auto ClassificationEERScorer::compute_auc(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int /*n_classes*/) const -> double
{
    const auto trials = build_gi_trials(embeddings, labels, 1U);
    return mann_whitney_auc(trials);
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
    return eer_from_trials(std::move(trials));
}

auto GenuineImpostorEERScorer::compute_auc(const std::vector<std::vector<float>>& embeddings,
    const std::vector<int>& labels,
    int /*n_classes*/) const -> double
{
    const auto trials = build_gi_trials(embeddings, labels, n_enroll_);
    return mann_whitney_auc(trials);
}

} // namespace statistics
