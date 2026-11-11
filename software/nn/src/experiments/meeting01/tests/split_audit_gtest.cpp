// split_audit_gtest.cpp — Leakage-safety audit for the nested LOSO fold assignment.
//
// Exercises meeting01::assign_speaker_fold on synthetic window metadata (no WAV
// corpus needed). The invariant under test: for every fold, the train / val / test
// window sets are disjoint AND speaker-disjoint AND recording-disjoint, and across
// all folds each speaker is the test speaker exactly once and the validation
// speaker exactly once (rotating validation).

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../lib/include/Meeting01Dataset.hpp"

using meeting01::assign_speaker_fold;
using meeting01::cap_windows_per_recording;
using meeting01::checkpoint_hash;
using meeting01::DatasetSplit;
using meeting01::SpeakerFoldAssignment;
using meeting01::split_composition_json;
using meeting01::stratified_window_cap;
using nn::dataLoaders::fsdd::WindowMetadata;

namespace
{

// K speakers, R recordings each, W windows per recording. recording_id and
// window_id are globally unique and ascending, mirroring FsddWindowDataset.
auto make_meta(int speakers, int recordings_per_speaker, int windows_per_recording)
    -> std::vector<WindowMetadata>
{
    std::vector<WindowMetadata> meta;
    int recording_id = 0;
    int window_id = 0;
    for (int s = 0; s < speakers; ++s)
    {
        const std::string name = "spk" + std::string(1, static_cast<char>('a' + s));
        for (int r = 0; r < recordings_per_speaker; ++r)
        {
            for (int w = 0; w < windows_per_recording; ++w)
            {
                // valid_length is irrelevant to fold-assignment leakage checks (this file
                // never computes a mask/loss) -- 256 is an arbitrary "full window" filler.
                meta.push_back(WindowMetadata{name, s, recording_id, window_id, w, w % 10, 256});
                ++window_id;
            }
            ++recording_id;
        }
    }
    return meta;
}

auto recordings_of(const std::vector<WindowMetadata>& meta, const std::vector<std::size_t>& idx)
    -> std::set<int>
{
    std::set<int> out;
    for (std::size_t i : idx) out.insert(meta[i].recording_id);
    return out;
}

auto speakers_of(const std::vector<WindowMetadata>& meta, const std::vector<std::size_t>& idx)
    -> std::set<int>
{
    std::set<int> out;
    for (std::size_t i : idx) out.insert(meta[i].speaker_id);
    return out;
}

template <typename T>
auto disjoint(const std::set<T>& a, const std::set<T>& b) -> bool
{
    for (const auto& x : a)
        if (b.count(x)) return false;
    return true;
}

} // namespace

TEST(SplitAudit, EveryFoldIsSpeakerRecordingAndWindowDisjoint)
{
    constexpr int kK = 6;
    const auto meta = make_meta(kK, /*recordings*/ 4, /*windows*/ 3);

    std::vector<int> test_speaker_seen(kK, 0);
    std::vector<int> val_speaker_seen(kK, 0);

    for (int fold = 0; fold < kK; ++fold)
    {
        const SpeakerFoldAssignment a = assign_speaker_fold(meta, fold, kK);

        // Window-index partition: disjoint and total.
        std::set<std::size_t> all;
        all.insert(a.train_idx.begin(), a.train_idx.end());
        all.insert(a.val_idx.begin(), a.val_idx.end());
        all.insert(a.test_idx.begin(), a.test_idx.end());
        EXPECT_EQ(all.size(), a.train_idx.size() + a.val_idx.size() + a.test_idx.size())
            << "fold " << fold << ": window index sets overlap";
        EXPECT_EQ(all.size(), meta.size()) << "fold " << fold << ": windows lost or duplicated";

        // Speaker disjointness.
        const auto tr_s = speakers_of(meta, a.train_idx);
        const auto va_s = speakers_of(meta, a.val_idx);
        const auto te_s = speakers_of(meta, a.test_idx);
        EXPECT_EQ(te_s.size(), 1u);
        EXPECT_EQ(va_s.size(), 1u);
        EXPECT_TRUE(disjoint(tr_s, va_s)) << "fold " << fold;
        EXPECT_TRUE(disjoint(tr_s, te_s)) << "fold " << fold;
        EXPECT_TRUE(disjoint(va_s, te_s)) << "fold " << fold;

        // Recording disjointness (the pseudoreplication-relevant guarantee).
        const auto tr_r = recordings_of(meta, a.train_idx);
        const auto va_r = recordings_of(meta, a.val_idx);
        const auto te_r = recordings_of(meta, a.test_idx);
        EXPECT_TRUE(disjoint(tr_r, va_r)) << "fold " << fold;
        EXPECT_TRUE(disjoint(tr_r, te_r)) << "fold " << fold;
        EXPECT_TRUE(disjoint(va_r, te_r)) << "fold " << fold;

        // Rotating validation: test = fold, val = (fold + 1) mod K.
        EXPECT_EQ(*te_s.begin(), fold);
        EXPECT_EQ(*va_s.begin(), (fold + 1) % kK);
        test_speaker_seen[static_cast<std::size_t>(*te_s.begin())]++;
        val_speaker_seen[static_cast<std::size_t>(*va_s.begin())]++;

        EXPECT_EQ(a.train_speakers.size(), static_cast<std::size_t>(kK - 2));
    }

    for (int s = 0; s < kK; ++s)
    {
        EXPECT_EQ(test_speaker_seen[static_cast<std::size_t>(s)], 1) << "speaker " << s;
        EXPECT_EQ(val_speaker_seen[static_cast<std::size_t>(s)], 1) << "speaker " << s;
    }
}

TEST(SplitAudit, RejectsOutOfRangeFold)
{
    const auto meta = make_meta(6, 2, 2);
    EXPECT_THROW(assign_speaker_fold(meta, -1, 6), std::runtime_error);
    EXPECT_THROW(assign_speaker_fold(meta, 6, 6), std::runtime_error);
}

TEST(SplitAudit, RejectsSpeakerCountMismatch)
{
    const auto meta = make_meta(5, 2, 2);                              // 5 speakers ...
    EXPECT_THROW(assign_speaker_fold(meta, 0, 6), std::runtime_error); // ... but asked for 6
}

TEST(SplitAudit, RejectsDegenerateFoldCount)
{
    const auto meta = make_meta(2, 2, 2);
    EXPECT_THROW(assign_speaker_fold(meta, 0, 1), std::runtime_error);
    // K == 2 leaves no training speaker → empty-partition guard fires.
    EXPECT_THROW(assign_speaker_fold(meta, 0, 2), std::runtime_error);
}

// Reproduces the exact shape of the "AudioMNIST window degeneracy" bug
// (.wiki/Experiments/Meeting01.md): more recordings (50) than the cap (30) needs, 2
// candidate windows each. Before the seeded per-recording shuffle was added, the
// round-robin always took idxs.front() = the lowest source_window_index, so the cap
// was reached during the very first pass over by_rec and every kept window was index
// 0 — never the spoken digit for AudioMNIST specifically, just its near-silent
// recording lead-in.
TEST(StratifiedWindowCap, ShuffleDiversifiesKeptWindowIndex)
{
    constexpr int kRecordings = 50;
    constexpr int kWindowsPerRecording = 2;
    constexpr int kCap = 30;

    auto meta = make_meta(kRecordings, /*recordings_per_speaker*/ 1, kWindowsPerRecording);
    std::vector<nn::Tensor> samples;
    samples.reserve(meta.size());
    for (std::size_t i = 0; i < meta.size(); ++i) samples.push_back(nn::Tensor::zeros(1, 1));
    std::vector<int> labels(meta.size(), 0);

    stratified_window_cap(samples, meta, &labels, kCap, /*seed*/ 7);

    ASSERT_EQ(meta.size(), static_cast<std::size_t>(kCap));
    ASSERT_EQ(samples.size(), meta.size());
    ASSERT_EQ(labels.size(), meta.size());

    const bool all_index_zero = std::all_of(meta.begin(),
        meta.end(),
        [](const WindowMetadata& m) { return m.source_window_index == 0; });
    EXPECT_FALSE(all_index_zero) << "every kept window is source_window_index 0 -- the "
                                    "AudioMNIST window degeneracy bug is back";

    // Per-recording fairness is unchanged by the shuffle: no recording contributes
    // more than one window when there are more recordings than the cap needs.
    std::set<int> rec_ids;
    for (const auto& m : meta) rec_ids.insert(m.recording_id);
    EXPECT_EQ(rec_ids.size(), static_cast<std::size_t>(kCap));
}

// A pool sorted the way FSDD's file list is: by digit first, so a cap that walks the list keeps
// the low digits only. `speakers` speakers, 10 digits, `trials` recordings of each, `windows`
// windows per recording.
static auto make_sorted_digit_pool(int speakers, int trials, int windows)
    -> std::vector<WindowMetadata>
{
    std::vector<WindowMetadata> meta;
    int recording = 0;
    int window_id = 0;
    for (int digit = 0; digit < 10; ++digit)
        for (int speaker = 0; speaker < speakers; ++speaker)
            for (int trial = 0; trial < trials; ++trial, ++recording)
                for (int w = 0; w < windows; ++w)
                    meta.push_back(WindowMetadata{"s" + std::to_string(speaker),
                        speaker,
                        recording,
                        window_id++,
                        w,
                        digit,
                        256});
    return meta;
}

static auto zero_samples(std::size_t n) -> std::vector<nn::Tensor>
{
    std::vector<nn::Tensor> samples;
    samples.reserve(n);
    for (std::size_t i = 0; i < n; ++i) samples.push_back(nn::Tensor::zeros(1, 1));
    return samples;
}

template <typename Key>
static auto count_by(const std::vector<WindowMetadata>& meta, Key key) -> std::map<int, int>
{
    std::map<int, int> counts;
    for (const auto& m : meta) ++counts[key(m)];
    return counts;
}

static void expect_within(const std::map<int, int>& counts, int spread, const char* what)
{
    const auto [low, high] = std::minmax_element(counts.begin(),
        counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    EXPECT_LE(high->second - low->second, spread) << what;
}

// FSDD fold 0 before 2026-11-04: 4 speakers, 50 trials per digit, 1000 windows of a ~13000 pool
// kept digits 0-4 only, and 150 of ~6500 digits 0-2, with the right window count both times.
TEST(StratifiedWindowCap, ASortedPoolKeepsEveryDigitAndSpeakerEvenly)
{
    for (const int cap : {1000, 150})
    {
        auto meta = make_sorted_digit_pool(/*speakers=*/4, /*trials=*/50, /*windows=*/6);
        auto samples = zero_samples(meta.size());
        stratified_window_cap(samples, meta, nullptr, cap, /*seed=*/44);

        ASSERT_EQ(meta.size(), static_cast<std::size_t>(cap));
        const auto digits = count_by(meta, [](const WindowMetadata& m) { return m.digit; });
        EXPECT_EQ(digits.size(), 10U) << "cap " << cap;
        expect_within(digits, 4, "digits");
        const auto speakers = count_by(meta, [](const WindowMetadata& m) { return m.speaker_id; });
        EXPECT_EQ(speakers.size(), 4U);
        expect_within(speakers, 1, "speakers");
        // The cap reaches recordings all over each digit, not one digit's first trials.
        const auto trials =
            count_by(meta, [](const WindowMetadata& m) { return m.recording_id % 50; });
        EXPECT_GT(trials.size(), 10U);
    }
}

// EEGMMIDB's validation: more subjects than the cap needs per recording, no digit (-1). Every
// subject must appear; before, 11 of 18 did.
TEST(StratifiedWindowCap, EverySubjectOfASortedPoolAppearsWithoutDigits)
{
    std::vector<WindowMetadata> meta;
    int recording = 0;
    int window_id = 0;
    for (int subject = 0; subject < 18; ++subject)
        for (int run = 0; run < 14; ++run, ++recording)
            for (int w = 0; w < 40; ++w)
                meta.push_back(WindowMetadata{
                    "S" + std::to_string(subject), subject, recording, window_id++, w, -1, 256});
    auto samples = zero_samples(meta.size());
    stratified_window_cap(samples, meta, nullptr, 150, /*seed=*/43);

    ASSERT_EQ(meta.size(), 150U);
    const auto subjects = count_by(meta, [](const WindowMetadata& m) { return m.speaker_id; });
    EXPECT_EQ(subjects.size(), 18U);
    expect_within(subjects, 1, "subjects");
    const auto recordings = count_by(meta, [](const WindowMetadata& m) { return m.recording_id; });
    EXPECT_EQ(recordings.size(), 150U) << "one window per recording before any repeats";
}

TEST(StratifiedWindowCap, KeepsLabelsAlignedAndARoomyCapChangesNothing)
{
    auto meta = make_sorted_digit_pool(2, 3, 2);
    auto samples = zero_samples(meta.size());
    std::vector<int> labels;
    labels.reserve(meta.size());
    for (const auto& m : meta) labels.push_back(m.digit);

    auto untouched = meta;
    auto untouched_samples = zero_samples(meta.size());
    stratified_window_cap(untouched_samples, untouched, nullptr, 100000, 1);
    EXPECT_EQ(untouched.size(), meta.size());

    stratified_window_cap(samples, meta, &labels, 25, 7);
    ASSERT_EQ(labels.size(), 25U);
    for (std::size_t i = 0; i < meta.size(); ++i) EXPECT_EQ(labels[i], meta[i].digit);
    std::set<int> ids;
    for (const auto& m : meta) ids.insert(m.window_id);
    EXPECT_EQ(ids.size(), 25U) << "no window twice";
}

// EEGMMIDB kept the first 64 s of each run, AudioMNIST its windows 0 and 1.
TEST(CapWindowsPerRecording, DrawsAcrossTheRecordingNotItsStart)
{
    auto meta = make_sorted_digit_pool(1, 40, 60); // 400 recordings of 60 windows
    auto samples = zero_samples(meta.size());
    cap_windows_per_recording(samples, meta, 2, /*seed=*/45);

    ASSERT_EQ(meta.size(), 800U);
    ASSERT_EQ(samples.size(), meta.size());
    const auto per_recording =
        count_by(meta, [](const WindowMetadata& m) { return m.recording_id; });
    EXPECT_EQ(per_recording.size(), 400U);
    expect_within(per_recording, 0, "windows per recording");

    const auto indices =
        count_by(meta, [](const WindowMetadata& m) { return m.source_window_index; });
    EXPECT_EQ(indices.size(), 60U) << "every position of a recording can be drawn";
    int late = 0;
    for (const auto& [index, n] : indices)
        if (index >= 30) late += n;
    EXPECT_GT(late, 340) << "about half of the draws fall in a recording's second half";
    EXPECT_LT(late, 460);
}

TEST(CapWindowsPerRecording, IsDeterministicAndOffWithoutACap)
{
    const auto run = [](unsigned int seed, int cap)
    {
        auto meta = make_sorted_digit_pool(2, 2, 10);
        auto samples = zero_samples(meta.size());
        cap_windows_per_recording(samples, meta, cap, seed);
        std::vector<int> ids;
        ids.reserve(meta.size());
        for (const auto& m : meta) ids.push_back(m.window_id);
        return ids;
    };
    EXPECT_EQ(run(7, 3), run(7, 3));
    EXPECT_NE(run(7, 3), run(8, 3));
    EXPECT_EQ(run(7, 0).size(), 400U);
    EXPECT_EQ(run(7, 1000).size(), 400U) << "a recording under the cap keeps all its windows";
}

TEST(SplitComposition, CountsSpeakersDigitsAndWindowPositions)
{
    const std::vector<WindowMetadata> meta{{"george", 0, 0, 0, 0, 3, 256},
        {"george", 0, 0, 1, 1, 3, 256},
        {"lucas", 1, 1, 2, 0, 7, 9}};
    const auto j = split_composition_json(meta);
    EXPECT_EQ(j.at("speakers").at("george"), 2);
    EXPECT_EQ(j.at("speakers").at("lucas"), 1);
    EXPECT_EQ(j.at("digits").at("3"), 2);
    EXPECT_EQ(j.at("digits").at("7"), 1);
    EXPECT_EQ(j.at("source_window_index").at("0"), 2);
    EXPECT_EQ(j.at("source_window_index").at("1"), 1);
}

// A code change that alters the windows leaves the profile as it is: the stamp must see the data.
TEST(CheckpointHash, FollowsTheConfigurationAndEveryWindow)
{
    const auto make_split = []
    {
        DatasetSplit split;
        split.train_samples = {nn::Tensor::zeros(2, 1), nn::Tensor::zeros(2, 1)};
        split.train_meta = {{"a", 0, 0, 0, 0, 1, 2}, {"a", 0, 0, 1, 1, 1, 2}};
        split.val_samples = {nn::Tensor::zeros(2, 1)};
        split.val_meta = {{"b", 1, 1, 2, 0, 1, 2}};
        split.test_samples = {nn::Tensor::zeros(2, 1)};
        split.test_meta = {{"c", 2, 2, 3, 0, 1, 2}};
        return split;
    };
    const DatasetSplit base = make_split();
    const std::size_t h = checkpoint_hash(5U, base);
    EXPECT_EQ(h, checkpoint_hash(5U, make_split()));
    EXPECT_NE(h, checkpoint_hash(6U, base));

    DatasetSplit one_value = make_split();
    one_value.test_samples[0].at(nn::Index{1}) = 1e-3F;
    EXPECT_NE(h, checkpoint_hash(5U, one_value));

    DatasetSplit other_window = make_split();
    other_window.val_meta[0].window_id = 9;
    EXPECT_NE(h, checkpoint_hash(5U, other_window));

    DatasetSplit one_fewer = make_split();
    one_fewer.train_samples.pop_back();
    one_fewer.train_meta.pop_back();
    EXPECT_NE(h, checkpoint_hash(5U, one_fewer));
}

TEST(StratifiedWindowCap, DeterministicForFixedSeed)
{
    constexpr int kRecordings = 50;
    constexpr int kWindowsPerRecording = 2;
    constexpr int kCap = 30;

    auto run = [&](unsigned int seed) -> std::vector<int>
    {
        auto meta = make_meta(kRecordings, 1, kWindowsPerRecording);
        std::vector<nn::Tensor> samples;
        samples.reserve(meta.size());
        for (std::size_t i = 0; i < meta.size(); ++i) samples.push_back(nn::Tensor::zeros(1, 1));
        stratified_window_cap(samples, meta, nullptr, kCap, seed);
        std::vector<int> window_ids;
        window_ids.reserve(meta.size());
        for (const auto& m : meta) window_ids.push_back(m.window_id);
        return window_ids;
    };

    EXPECT_EQ(run(7), run(7));
    // Different seeds are not required to differ (small sample space), but should
    // still both diversify away from all-index-0 -- covered by the test above.
}
