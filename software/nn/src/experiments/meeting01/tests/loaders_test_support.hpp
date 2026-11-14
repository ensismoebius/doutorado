// loaders_test_support.hpp -- helpers shared by loaders_gtest (synthetic fixtures, always run)
// and loaders_corpus_gtest (the real corpora, opt-in). Header-only, inline.

#pragma once

#include <set>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01Dataset.hpp"

namespace meeting01_test
{

inline meeting01::Meeting01Config base_config()
{
    meeting01::Meeting01Config c;
    c.experiment.run_tag = "loaders_test";
    c.experiment.seed = 42;
    c.experiment.repeats = 1;
    c.dataset.window_size = 256;
    c.dataset.cv_num_folds = 6;
    c.dataset.results_dir = ""; // no manifest side effects
    return c;
}

inline int distinct_speakers(const std::vector<meeting01::WindowMetadata>& m)
{
    std::set<int> s;
    for (const auto& w : m) s.insert(w.speaker_id);
    return static_cast<int>(s.size());
}

} // namespace meeting01_test
