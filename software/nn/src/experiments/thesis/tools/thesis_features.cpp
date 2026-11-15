/**
 * @file src/experiments/thesis/tools/thesis_features.cpp
 * @brief Write the feature vectors a thesis profile computes, clean and degraded, one row per
 *        sample.
 *
 * WHY. Some questions are about the features themselves, not about an EER: "does
 * resynthesis move an utterance away from its own speaker?" is answered utterance by utterance
 * (1974 measures), while the EER pools them into one number per fold (15 measures, too few to
 * show a small effect: the identity pilot, proposal section 4.3). This tool runs exactly the
 * steps of the thesis binary up to the classifier (load_dataset, extract_features,
 * degrade_voices, extract_features again), so what it writes is what the experiment classified.
 *
 * Usage:
 *   thesis_features --config <profile.json> --out-dir <dir>
 * writes <dir>/<run_tag>_features_clean.csv and, when the profile enables a degradation,
 * <dir>/<run_tag>_features_degraded.csv. Columns: sample, subject_id, stimulus, text, then
 * f0..f<D-1> (full precision). An autoencoder strategy is refused: its features depend on a
 * training seed, and this tool runs no repeat.
 */

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DegradeVoices.hpp"
#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"

namespace
{

void write_csv(const std::string& path,
    const thesis::ThesisDatasetView& view,
    const std::vector<thesis::FeatureSet>& sets)
{
    if (sets.size() != 1)
        throw std::runtime_error("thesis_features: the profile yields " +
                                 std::to_string(sets.size()) +
                                 " feature sets; this tool writes exactly one");
    const auto& vectors = sets.front().vectors;
    if (vectors.size() != view.samples.size())
        throw std::logic_error("thesis_features: " + std::to_string(vectors.size()) +
                               " vectors for " + std::to_string(view.samples.size()) + " samples");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("thesis_features: cannot write " + path);
    const std::size_t dims = vectors.front().size();
    out << "sample,subject_id,stimulus,text";
    for (std::size_t d = 0; d < dims; ++d) out << ",f" << d;
    out << "\n";
    char buf[32];
    for (std::size_t i = 0; i < vectors.size(); ++i)
    {
        if (vectors[i].size() != dims)
            throw std::logic_error("thesis_features: sample " + std::to_string(i) + " has " +
                                   std::to_string(vectors[i].size()) + " features, not " +
                                   std::to_string(dims));
        const auto& s = view.samples[i];
        if (s.text_phrase.find_first_of(",\"\n") != std::string::npos)
            throw std::runtime_error("thesis_features: the text of sample " + std::to_string(i) +
                                     " contains a comma, quote or newline; the CSV would break");
        out << i << ',' << s.subject_id << ',' << s.stimulus << ',' << s.text_phrase;
        for (const double v : vectors[i])
        {
            std::snprintf(buf, sizeof buf, "%.17g", v);
            out << ',' << buf;
        }
        out << "\n";
    }
    std::cout << "[thesis_features] " << vectors.size() << " x " << dims << " -> " << path << "\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string config;
        std::string out_dir;
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "--config" && i + 1 < argc)
                config = argv[++i];
            else if (a == "--out-dir" && i + 1 < argc)
                out_dir = argv[++i];
            else
                throw std::invalid_argument("thesis_features: unknown argument \"" + a + "\"");
        }
        if (config.empty() || out_dir.empty())
            throw std::invalid_argument(
                "usage: thesis_features --config <profile.json> --out-dir <dir> (both required)");
        const auto cfg = thesis::ThesisConfig::from_file(config);
        if (cfg.feature_extraction.strategy.get() == "autoencoder")
            throw std::invalid_argument("thesis_features: " + config +
                                        " uses an autoencoder, whose features depend on a "
                                        "training seed; use a handcrafted or PCA profile");
        std::filesystem::create_directories(out_dir);
        const std::string stem = out_dir + "/" + cfg.experiment.run_tag.get();

        auto view = thesis::load_dataset(cfg.dataset);
        write_csv(stem + "_features_clean.csv",
            view,
            thesis::extract_features(view, cfg, /*on_model_trained=*/nullptr));
        if (cfg.degradation.enabled.declared() && cfg.degradation.enabled)
        {
            thesis::degrade_voices(view, cfg.degradation);
            write_csv(stem + "_features_degraded.csv",
                view,
                thesis::extract_features(view, cfg, /*on_model_trained=*/nullptr));
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[thesis_features] ERROR: " << e.what() << "\n";
        return 1;
    }
}
