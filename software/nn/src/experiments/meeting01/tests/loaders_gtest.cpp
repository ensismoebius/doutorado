// loaders_gtest.cpp -- fast checks of the EDF-format EEG loader (shared by eegmmidb and siena)
// on hand-built fixtures: header parsing, digital decoding, subject grouping, resampling to the
// dataset rate. They write their own files into a temp directory, so they need no corpus and
// always run (CI included).
//
// The real-corpus checks (FSDD, AudioMNIST, MIT-BIH, eegmmidb, Siena) are in
// loaders_corpus_gtest.cpp: they are registered only when -DNN_TEST_DB_ROOT=<dir> is given at
// configure time, and fail (never skip) when a corpus they need is missing.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include "Meeting01Eeg.hpp"
#include "loaders_test_support.hpp"

namespace
{
namespace fs = std::filesystem;
using meeting01_test::distinct_speakers;

std::string edf_field(const std::string& value, std::size_t width)
{
    if (value.size() >= width) return value.substr(0, width);
    return value + std::string(width - value.size(), ' ');
}

// Writes a minimal single-signal EDF file: header fields sized exactly per the
// spec (https://www.edfplus.info/specs/edf.html), one data record per
// samples_per_record-sized chunk of `digital_values`, little-endian int16
// samples. physical_min/max and digital_min/max are chosen so scale == 1 and
// offset == 0 (physical == digital exactly), which keeps the test's expected
// values simple without needing to special-case the conversion formula.
void write_synthetic_edf(const std::filesystem::path& path,
    const std::vector<std::int16_t>& digital_values,
    int samples_per_record,
    const std::string& record_seconds = "1")
{
    const int n_records = static_cast<int>(digital_values.size()) / samples_per_record;
    ASSERT_EQ(static_cast<int>(digital_values.size()), n_records * samples_per_record);

    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open());

    const int header_bytes = 256 + 256; // main header + 1 signal's per-signal header
    out << edf_field("0", 8);           // version
    out << edf_field("synthetic patient", 80);
    out << edf_field("synthetic recording", 80);
    out << edf_field("01.01.01", 8); // startdate
    out << edf_field("00.00.00", 8); // starttime
    out << edf_field(std::to_string(header_bytes), 8);
    out << edf_field("", 44); // reserved
    out << edf_field(std::to_string(n_records), 8);
    out << edf_field(record_seconds, 8); // duration of a data record, seconds
    out << edf_field("1", 4);            // ns

    out << edf_field("EEG synth", 16); // label
    out << edf_field("", 80);          // transducer
    out << edf_field("uV", 8);         // physical dimension
    out << edf_field("-16384", 8);     // physical_min
    out << edf_field("16383", 8);      // physical_max
    out << edf_field("-16384", 8);     // digital_min
    out << edf_field("16383", 8);      // digital_max (scale == 1, offset == 0)
    out << edf_field("", 80);          // prefiltering
    out << edf_field(std::to_string(samples_per_record), 8);
    out << edf_field("", 32); // reserved

    out.write(reinterpret_cast<const char*>(digital_values.data()),
        static_cast<std::streamsize>(digital_values.size() * sizeof(std::int16_t)));
}

} // namespace

TEST(Meeting01Loaders, EegSyntheticEdfDecodesAndGroupsBySubject)
{
    const fs::path root = fs::temp_directory_path() / "meeting01_eeg_synth_test";
    fs::remove_all(root);
    fs::create_directories(root / "subjA");
    fs::create_directories(root / "subjB");

    // Monotonically increasing digital ramp; physical_min/max == digital_min/max in
    // write_synthetic_edf, so scale == 1 and physical == digital exactly. This test exercises
    // decode/windowing/subject-grouping mechanics only -- it deliberately does NOT assert
    // per-sample monotonicity survives z-score any more, because the bandpass+notch filter now
    // runs first: a monotonic ramp IS low-frequency/DC content by construction, so a 0.5 Hz
    // highpass correctly removes almost all of it. That is the filter doing its job, not a
    // regression -- see BandpassNotchFilter.SubLowCutoffDriftIsAttenuated for the dedicated
    // frequency-response test.
    std::vector<std::int16_t> ramp(16);
    for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<std::int16_t>(i);
    // 16 samples in one record of 1/32 s: 512 Hz, the rate the dataset below asks for.
    write_synthetic_edf(root / "subjA" / "subjA_01.edf", ramp, 16, "0.03125");
    write_synthetic_edf(root / "subjB" / "subjB_01.edf", ramp, 16, "0.03125");

    meeting01::EegWindowDataset ds(root,
        /*window_size=*/8,
        /*sampling_rate=*/512.0,
        /*notch_hz=*/-1.0);
    ASSERT_EQ(ds.size(), 4u); // 2 windows/file x 2 files
    EXPECT_EQ(ds.windows().size(), ds.metadata().size());
    EXPECT_EQ(distinct_speakers(ds.metadata()), 2);

    for (std::size_t i = 0; i < ds.windows().size(); ++i)
    {
        const auto& w = ds.windows()[i];
        EXPECT_EQ(w.size(), 8);

        const auto& m = ds.metadata()[i];
        EXPECT_TRUE(m.speaker == "subjA" || m.speaker == "subjB");
        EXPECT_EQ(m.source_window_index, static_cast<int>(i) % 2);
        EXPECT_EQ(m.valid_length, 8); // EEG windowing never pads a trailing partial window
    }

    fs::remove_all(root);
}

// S088, S092 and S100 of EEGMMIDB are at 128 Hz among files at 160 Hz. The same 10 Hz tone at
// both rates must give the same windows: same count, same duration, same values. Treated as
// 160 Hz, as they were until 2026-11-04, the 128 Hz windows held a 12.5 Hz tone and the file
// fewer of them.
TEST(Meeting01Loaders, EegFilesAtAnotherRateBecomeTheDatasetRate)
{
    const fs::path root = fs::temp_directory_path() / "meeting01_eeg_rates_test";
    fs::remove_all(root);
    fs::create_directories(root / "S001");
    fs::create_directories(root / "S088");

    constexpr int kSeconds = 41; // 128 Hz: the last sample falls 8 ms short of the end
    const auto tone = [](int rate)
    {
        std::vector<std::int16_t> x(static_cast<std::size_t>(rate * kSeconds));
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<std::int16_t>(std::lround(
                8000.0 * std::sin(2.0 * std::numbers::pi * 10.0 * static_cast<double>(i) / rate)));
        return x;
    };
    write_synthetic_edf(root / "S001" / "S001R01.edf", tone(160), 160);
    write_synthetic_edf(root / "S088" / "S088R01.edf", tone(128), 128);

    const meeting01::EegWindowDataset ds(root, 256, /*sampling_rate=*/160.0, /*notch_hz=*/60.0);
    EXPECT_EQ(ds.resampled_recordings(), 1);

    std::vector<std::size_t> a;
    std::vector<std::size_t> b;
    for (std::size_t i = 0; i < ds.size(); ++i)
        (ds.metadata()[i].speaker == "S001" ? a : b).push_back(i);
    ASSERT_EQ(a.size(), 25U);
    ASSERT_EQ(b.size(), a.size());

    // Windows 8-16: well inside the recording, away from the filter's edges.
    for (std::size_t w = 8; w <= 16; ++w)
    {
        const auto& x = ds.windows()[a[w]];
        const auto& y = ds.windows()[b[w]];
        double squared = 0.0;
        for (nn::Index t = 0; t < 256; ++t)
        {
            const double d = static_cast<double>(x.at(t, 0)) - static_cast<double>(y.at(t, 0));
            squared += d * d;
        }
        EXPECT_LT(squared / 256.0, 1e-3) << "window " << w;
    }
    fs::remove_all(root);
}

TEST(Meeting01Loaders, EegRefusesARecordOfNoDurationAndARateBelowTheBand)
{
    const fs::path root = fs::temp_directory_path() / "meeting01_eeg_bad_header_test";
    fs::remove_all(root);
    fs::create_directories(root / "S001");
    const std::vector<std::int16_t> samples(320, 1);
    write_synthetic_edf(root / "S001" / "S001R01.edf", samples, 160, "0");
    EXPECT_THROW(
        { [[maybe_unused]] const meeting01::EegWindowDataset object(root, 256, 160.0, 60.0); },
        std::runtime_error);
    EXPECT_THROW(
        { [[maybe_unused]] const meeting01::EegWindowDataset object(root, 256, 64.0, 60.0); },
        std::invalid_argument);
    fs::remove_all(root);
}
