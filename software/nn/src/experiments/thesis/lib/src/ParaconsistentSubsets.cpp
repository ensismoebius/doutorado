#include "ParaconsistentSubsets.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

#include "ThesisParaconsistent.hpp"
#include "statistics/GroupKFoldPolicy.hpp"

namespace thesis
{

namespace
{

void score_subset(const std::string& name,
    const std::vector<std::size_t>& keep,
    const std::vector<ThesisSample>& samples,
    const std::vector<FeatureSet>& feature_sets,
    std::vector<SubsetScore>& out)
{
    if (keep.empty())
        throw std::logic_error("ParaconsistentSubsets: subset " + name + " keeps no sample");
    std::vector<ThesisSample> sub_samples;
    sub_samples.reserve(keep.size());
    for (std::size_t i : keep)
    {
        ThesisSample s; // labels only: the score never reads the signals
        s.subject_id = samples[i].subject_id;
        s.stimulus = samples[i].stimulus;
        s.text_phrase = samples[i].text_phrase;
        sub_samples.push_back(std::move(s));
    }
    for (const auto& fs : feature_sets)
    {
        FeatureSet sub{fs.label, {}};
        sub.vectors.reserve(keep.size());
        for (std::size_t i : keep) sub.vectors.push_back(fs.vectors.at(i));
        const ParaconsistentScore p = score_feature_set(sub_samples, sub);
        out.push_back({name, fs.label, p.alpha, p.beta, p.d_truth, p.d_penalized});
    }
}

} // namespace

auto score_speaker_subsets(const std::vector<ThesisSample>& samples,
    const std::vector<FeatureSet>& feature_sets,
    int fold_k,
    std::uint32_t fold_seed,
    int fold_repeats) -> std::vector<SubsetScore>
{
    if (fold_k < 2 || fold_repeats < 1 || fold_seed == 0u)
        throw std::invalid_argument(
            "ParaconsistentSubsets: need fold_k >= 2, fold_repeats >= 1 "
            "and a non-zero fold_seed");
    std::vector<int> groups;
    groups.reserve(samples.size());
    for (const auto& s : samples) groups.push_back(s.subject_id);
    const std::set<int> speakers(groups.begin(), groups.end());
    if (speakers.size() < 3)
        throw std::invalid_argument("ParaconsistentSubsets: fewer than 3 speakers");

    std::vector<SubsetScore> out;
    for (int left_out : speakers)
    {
        std::vector<std::size_t> keep;
        for (std::size_t i = 0; i < samples.size(); ++i)
            if (groups[i] != left_out) keep.push_back(i);
        score_subset("loso:" + std::to_string(left_out), keep, samples, feature_sets, out);
    }
    for (int r = 0; r < fold_repeats; ++r)
    {
        // Phase 01 repetition r runs with seed + r (thesis.cpp), and its flat and nested
        // outer folds are GroupKFoldPolicy(k, shuffle = true, that seed).
        const statistics::GroupKFoldPolicy policy(
            static_cast<std::size_t>(fold_k), true, fold_seed + static_cast<std::uint32_t>(r));
        const auto splits = policy.make_splits(samples.size(), groups);
        for (std::size_t f = 0; f < splits.size(); ++f)
            score_subset("fold:r" + std::to_string(r) + "f" + std::to_string(f),
                splits[f].train_indices,
                samples,
                feature_sets,
                out);
    }
    return out;
}

void write_subsets_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<SubsetScore>& scores)
{
    std::filesystem::create_directories(results_dir);
    const std::string path = results_dir + "/e05_" + run_tag + "_paraconsistent_subsets.csv";
    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("ParaconsistentSubsets: cannot write " + path);
    f << "subset,label,alpha,beta,d_truth,d_penalized\n";
    f << std::fixed << std::setprecision(8);
    for (const auto& s : scores)
        f << s.subset << "," << s.label << "," << s.alpha << "," << s.beta << "," << s.d_truth
          << "," << s.d_penalized << "\n";
    if (!f) throw std::runtime_error("ParaconsistentSubsets: writing " + path + " failed");
}

} // namespace thesis
