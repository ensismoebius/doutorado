// split_audit_gtest.cpp — Leakage-safety audit for the nested LOSO fold assignment.
//
// Exercises meeting01::assign_speaker_fold on synthetic window metadata (no WAV
// corpus needed). The invariant under test: for every fold, the train / val / test
// window sets are disjoint AND speaker-disjoint AND recording-disjoint, and across
// all folds each speaker is the test speaker exactly once and the validation
// speaker exactly once (rotating validation).

#include <gtest/gtest.h>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "../lib/include/Meeting01Dataset.hpp"

using meeting01::assign_speaker_fold;
using meeting01::SpeakerFoldAssignment;
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
                meta.push_back(WindowMetadata{name, s, recording_id, window_id, w, w % 10});
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
