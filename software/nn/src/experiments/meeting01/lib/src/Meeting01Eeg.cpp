// Meeting01Eeg.cpp — see Meeting01Eeg.hpp.
//
// EDF header layout verified against https://www.edfplus.info/specs/edf.html
// (Kemp et al. 1992) 2026-09-23: fixed 256-byte main header (8/80/80/8/8/8/44/8/8/4
// ascii fields, left-justified, space-padded), then ns * 256 bytes of per-signal
// fields (16/80/8/8/8/8/8/80/8/32 ascii each, laid out block-per-field-type — all ns
// labels first, then all ns transducer types, ... — not interleaved per signal).
// Samples are 2-byte little-endian two's-complement integers (confirmed via the
// EDF+ spec, https://www.edfplus.info/specs/edfplus.html, additional spec #7), one
// signal's full per-record sample block after another within each data record.
//
// Digital -> physical conversion is the standard EDF linear mapping. Neither spec
// page prints the formula as an equation (only the prose "these 4 extreme values
// specify offset and amplification"); the formula below was cross-checked against
// the spec page's own worked example (physical_min=-440, physical_max=510,
// digital_min=-2048, digital_max=2047 -> digital=0 yields physical=35.1 uV,
// matching the page's stated 35 uV offset; slope 950/4095=0.232 uV/count inverts to
// the page's stated 4.31 counts/uV gain) — measured against that example, not
// assumed from memory alone.
#include "Meeting01Eeg.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>

#include "utility/SignalPreprocessing.hpp"

namespace meeting01
{
namespace
{

std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

std::string read_field(std::ifstream& in, int width, const std::string& edf_path)
{
    std::string buf(static_cast<std::size_t>(width), '\0');
    in.read(buf.data(), width);
    if (!in)
        throw std::runtime_error(
            "EegWindowDataset: unexpected EOF reading EDF header field in " + edf_path);
    return trim(buf);
}

struct EdfHeader
{
    int header_bytes = 0;
    int n_signals = 0;
    std::vector<double> physical_min;
    std::vector<double> physical_max;
    std::vector<double> digital_min;
    std::vector<double> digital_max;
    std::vector<int> samples_per_record;
};

EdfHeader parse_edf_header(std::ifstream& in, const std::string& edf_path)
{
    auto field = [&](int width) { return read_field(in, width, edf_path); };

    EdfHeader h;
    (void) field(8);  // version
    (void) field(80); // patient id
    (void) field(80); // recording id
    (void) field(8);  // startdate
    (void) field(8);  // starttime
    h.header_bytes = std::stoi(field(8));
    (void) field(44); // reserved
    (void) field(8);  // number of data records (unreliable — may be -1; we read to EOF instead)
    (void) field(8);  // duration of a data record, in seconds
    h.n_signals = std::stoi(field(4));

    if (h.n_signals <= 0)
        throw std::runtime_error("EegWindowDataset: EDF header declares 0 signals in " + edf_path);

    const int ns = h.n_signals;
    for (int i = 0; i < ns; ++i) (void) field(16); // label
    for (int i = 0; i < ns; ++i) (void) field(80); // transducer type
    for (int i = 0; i < ns; ++i) (void) field(8);  // physical dimension

    h.physical_min.resize(static_cast<std::size_t>(ns));
    for (auto& v : h.physical_min) v = std::stod(field(8));
    h.physical_max.resize(static_cast<std::size_t>(ns));
    for (auto& v : h.physical_max) v = std::stod(field(8));
    h.digital_min.resize(static_cast<std::size_t>(ns));
    for (auto& v : h.digital_min) v = std::stod(field(8));
    h.digital_max.resize(static_cast<std::size_t>(ns));
    for (auto& v : h.digital_max) v = std::stod(field(8));

    for (int i = 0; i < ns; ++i) (void) field(80); // prefiltering

    h.samples_per_record.resize(static_cast<std::size_t>(ns));
    for (auto& v : h.samples_per_record) v = std::stoi(field(8));

    for (int i = 0; i < ns; ++i) (void) field(32); // reserved

    // Self-check: the fixed 256-byte main header plus ns * 256 per-signal bytes
    // must equal the header's own declared size. A mismatch means either this
    // parser's field widths are wrong for this file, or the file is not a
    // standard EDF/EDF+ file — either way, better to throw than to silently
    // start reading signal data from the wrong offset.
    const auto pos = in.tellg();
    if (pos != static_cast<std::streamoff>(h.header_bytes))
        throw std::runtime_error("EegWindowDataset: EDF header size mismatch in " + edf_path +
                                 " (header declares " + std::to_string(h.header_bytes) +
                                 " bytes, parser consumed " + std::to_string(pos) + ")");

    return h;
}

// Reads signal 0's full time series (physical units) from one .edf file.
std::vector<float> read_signal0(const std::filesystem::path& edf)
{
    std::ifstream in(edf, std::ios::binary);
    if (!in) throw std::runtime_error("EegWindowDataset: cannot open " + edf.string());

    const EdfHeader h = parse_edf_header(in, edf.string());
    if (h.digital_max[0] == h.digital_min[0])
        throw std::runtime_error(
            "EegWindowDataset: signal 0 has digital_max == digital_min in " + edf.string());

    const double scale =
        (h.physical_max[0] - h.physical_min[0]) / (h.digital_max[0] - h.digital_min[0]);
    const double offset = h.physical_min[0] - h.digital_min[0] * scale;

    const int s0_per_record = h.samples_per_record[0];
    long other_bytes_per_record = 0;
    for (std::size_t i = 1; i < h.samples_per_record.size(); ++i)
        other_bytes_per_record += static_cast<long>(h.samples_per_record[i]) * 2;

    // n_data_records in the header is often -1 (unknown, e.g. for a live/streamed
    // recording) — read records until a short read rather than trusting it.
    std::vector<float> out;
    std::vector<std::int16_t> buf(static_cast<std::size_t>(s0_per_record));
    while (true)
    {
        in.read(
            reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(s0_per_record) * 2);
        if (in.gcount() != static_cast<std::streamsize>(s0_per_record) * 2) break;
        for (std::int16_t v : buf) out.push_back(static_cast<float>(v * scale + offset));

        if (other_bytes_per_record > 0)
        {
            in.seekg(other_bytes_per_record, std::ios::cur);
            if (!in) break;
        }
    }
    return out;
}

} // namespace

EegWindowDataset::EegWindowDataset(const std::filesystem::path& dataset_root, int window_size)
{
    if (window_size <= 0) throw std::invalid_argument("EegWindowDataset: window_size must be > 0");
    if (!std::filesystem::exists(dataset_root))
        throw std::runtime_error("EegWindowDataset: root does not exist: " + dataset_root.string());

    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::recursive_directory_iterator(dataset_root))
        if (e.is_regular_file() && e.path().extension() == ".edf") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("EegWindowDataset: no .edf files under " + dataset_root.string());

    // Dense subject_id over the sorted set of distinct parent-directory names.
    std::map<std::string, int> subject_id;
    for (const auto& f : files) subject_id.emplace(f.parent_path().filename().string(), 0);
    {
        int next = 0;
        for (auto& kv : subject_id) kv.second = next++;
    }

    int global_window_id = 0;
    for (std::size_t rec = 0; rec < files.size(); ++rec)
    {
        const std::vector<float> sig0 = read_signal0(files[rec]);
        const std::string subject = files[rec].parent_path().filename().string();

        int source_window_idx = 0;
        for (std::size_t offset = 0; offset + static_cast<std::size_t>(window_size) <= sig0.size();
            offset += static_cast<std::size_t>(window_size))
        {
            nn::Tensor window(static_cast<nn::Index>(window_size), 1);
            for (int t = 0; t < window_size; ++t)
                window.at(t, 0) = sig0[offset + static_cast<std::size_t>(t)];
            nn::utility::zscore_inplace(window);

            windows_.push_back(std::move(window));
            labels_.push_back(-1);
            metadata_.push_back(WindowMetadata{
                subject,
                subject_id.at(subject),
                static_cast<int>(rec),
                global_window_id,
                source_window_idx,
                -1,
            });
            ++global_window_id;
            ++source_window_idx;
        }
    }

    if (windows_.empty())
        throw std::runtime_error("EegWindowDataset: produced no windows (window_size " +
                                 std::to_string(window_size) + " larger than every recording?)");
}

} // namespace meeting01
