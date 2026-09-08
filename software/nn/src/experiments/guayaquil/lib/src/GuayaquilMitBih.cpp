// GuayaquilMitBih.cpp — see GuayaquilMitBih.hpp.

#include "GuayaquilMitBih.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "utility/SignalPreprocessing.hpp"

namespace guayaquil
{
namespace
{

struct WfdbHeader
{
    std::string record;
    int nsig = 0;
    long nsamp = 0;
    std::string dat_file; // signal 0's data file
    int format = 0;       // signal 0's storage format
    double gain = 0.0;    // adu per physical unit
    int adc_zero = 0;     // adu value of physical zero
};

// Parse the leading floating-point prefix of a WFDB gain field such as
// "200", "200(0)/mV" or "200/mV". Returns 0.0 if there is no numeric prefix.
auto leading_double(const std::string& s) -> double
{
    std::size_t i = 0;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) != 0 || s[i] == '.' ||
                               s[i] == '-' || s[i] == '+'))
        ++i;
    if (i == 0) return 0.0;
    try
    {
        return std::stod(s.substr(0, i));
    }
    catch (...)
    {
        return 0.0;
    }
}

auto parse_header(const std::filesystem::path& hea) -> WfdbHeader
{
    std::ifstream in(hea);
    if (!in) throw std::runtime_error("MitBihWindowDataset: cannot open header " + hea.string());

    WfdbHeader h;
    std::string line;

    // Record line: "<name> <nsig> <fs[/counts]> <nsamp> ..."
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string fs_tok;
        ss >> h.record >> h.nsig >> fs_tok >> h.nsamp;
        break;
    }
    if (h.nsig <= 0)
        throw std::runtime_error("MitBihWindowDataset: bad record line in " + hea.string());

    // First signal specification line.
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string gain_tok;
        int adc_res = 0;
        ss >> h.dat_file >> h.format >> gain_tok >> adc_res >> h.adc_zero;
        h.gain = leading_double(gain_tok);
        break;
    }
    if (h.format != 212)
        throw std::runtime_error("MitBihWindowDataset: only format 212 is supported, header " +
                                 hea.string() + " declares format " + std::to_string(h.format));
    if (h.gain == 0.0)
        throw std::runtime_error("MitBihWindowDataset: header " + hea.string() +
                                 " has gain 0 — cannot convert to physical units.");
    return h;
}

// Decode all 12-bit two's-complement samples from a format-212 stream.
auto decode_212(const std::filesystem::path& dat) -> std::vector<int>
{
    std::ifstream in(dat, std::ios::binary);
    if (!in) throw std::runtime_error("MitBihWindowDataset: cannot open data file " + dat.string());
    std::vector<unsigned char> bytes(
        (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() % 3 != 0)
        throw std::runtime_error(
            "MitBihWindowDataset: format-212 file size not a multiple of 3: " + dat.string());

    std::vector<int> out;
    out.reserve(bytes.size() / 3 * 2);
    for (std::size_t i = 0; i + 2 < bytes.size(); i += 3)
    {
        const int b0 = bytes[i];
        const int b1 = bytes[i + 1];
        const int b2 = bytes[i + 2];
        int first = ((b1 & 0x0F) << 8) | b0;
        int second = (b2 << 4) | ((b1 >> 4) & 0x0F);
        if (first >= 2048) first -= 4096;
        if (second >= 2048) second -= 4096;
        out.push_back(first);
        out.push_back(second);
    }
    return out;
}

} // namespace

MitBihWindowDataset::MitBihWindowDataset(const std::filesystem::path& dataset_root, int window_size)
{
    if (window_size <= 0)
        throw std::invalid_argument("MitBihWindowDataset: window_size must be > 0");
    if (!std::filesystem::exists(dataset_root))
        throw std::runtime_error(
            "MitBihWindowDataset: root does not exist: " + dataset_root.string());

    // Non-recursive: mitdb keeps its 48 records at the root; re-annotation sets
    // (x_mitdb/, mitdbdir/) live in subdirectories and must not be picked up.
    std::vector<std::filesystem::path> headers;
    for (const auto& e : std::filesystem::directory_iterator(dataset_root))
        if (e.is_regular_file() && e.path().extension() == ".hea") headers.push_back(e.path());
    std::sort(headers.begin(), headers.end());
    if (headers.empty())
        throw std::runtime_error(
            "MitBihWindowDataset: no .hea files under " + dataset_root.string());

    // Dense speaker_id over the sorted record names (one record == one group).
    std::map<std::string, int> record_id;
    for (const auto& hea : headers) record_id.emplace(hea.stem().string(), 0);
    {
        int next = 0;
        for (auto& kv : record_id) kv.second = next++;
    }

    int global_window_id = 0;
    for (std::size_t rec = 0; rec < headers.size(); ++rec)
    {
        const WfdbHeader h = parse_header(headers[rec]);
        const std::filesystem::path dat = headers[rec].parent_path() / h.dat_file;
        const std::vector<int> flat = decode_212(dat);

        // Channel-interleaved: signal 0 is every h.nsig-th value from offset 0.
        const std::size_t n_per_sig = flat.size() / static_cast<std::size_t>(h.nsig);
        std::vector<float> sig0;
        sig0.reserve(n_per_sig);
        for (std::size_t i = 0; i < n_per_sig; ++i)
            sig0.push_back(static_cast<float>(
                (flat[i * static_cast<std::size_t>(h.nsig)] - h.adc_zero) / h.gain));

        const std::string rec_name = headers[rec].stem().string();
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
                rec_name,
                record_id.at(rec_name),
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
        throw std::runtime_error("MitBihWindowDataset: produced no windows (window_size " +
                                 std::to_string(window_size) + " larger than every record?)");
}

} // namespace guayaquil
