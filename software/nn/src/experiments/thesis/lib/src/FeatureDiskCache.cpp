/**
 * @file src/experiments/thesis/lib/src/FeatureDiskCache.cpp
 * @brief FeatureDiskCache.hpp.
 */
#include "FeatureDiskCache.hpp"

#include <unistd.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "progress/ProgressManager.hpp"

namespace thesis
{

namespace
{

// Last byte = format version.
constexpr char kMagic[8] = {'E', '0', '5', 'F', 'C', '\0', '\0', '\1'};

struct Fnv
{
    std::uint64_t h = 1469598103934665603ULL;
    void bytes(const void* data, std::size_t n)
    {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    template <typename T>
    void pod(const T& v)
    {
        bytes(&v, sizeof(T));
    }
    void str(const std::string& s)
    {
        pod(static_cast<std::uint64_t>(s.size()));
        bytes(s.data(), s.size());
    }
    void tensor(const nn::Tensor& t)
    {
        pod(static_cast<std::uint64_t>(t.rows()));
        pod(static_cast<std::uint64_t>(t.cols()));
        bytes(t.data_ptr(), static_cast<std::size_t>(t.rows() * t.cols()) * sizeof(float));
    }
};

auto hex(std::uint64_t v) -> std::string
{
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << v;
    return oss.str();
}

template <typename T>
void write_pod(std::ostream& os, const T& v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
auto read_pod(std::istream& is, T& v) -> bool
{
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&v), sizeof(T)));
}

void write_string(std::ostream& os, const std::string& s)
{
    write_pod(os, static_cast<std::uint64_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

auto read_string(std::istream& is, std::string& s) -> bool
{
    std::uint64_t n = 0;
    constexpr std::uint64_t kMaxString = 1073741824ULL; // 1 GiB: a larger length is corrupt
    if (!read_pod(is, n) || n > kMaxString) return false;
    s.resize(static_cast<std::size_t>(n));
    return n == 0 || static_cast<bool>(is.read(s.data(), static_cast<std::streamsize>(n)));
}

auto payload_checksum(const FeatureSet& fs) -> std::uint64_t
{
    Fnv f;
    f.str(fs.label);
    for (const auto& row : fs.vectors) f.bytes(row.data(), row.size() * sizeof(double));
    return f.h;
}

auto host_name() -> std::string
{
    std::array<char, 256> host{};
    if (::gethostname(host.data(), host.size() - 1) != 0)
        throw std::runtime_error("FeatureDiskCache: gethostname failed; cannot name a temp file");
    return host.data();
}

} // namespace

auto view_fingerprint(const ThesisDatasetView& view) -> std::uint64_t
{
    Fnv f;
    f.pod(static_cast<std::uint64_t>(view.samples.size()));
    f.pod(view.voice_sample_rate);
    f.pod(view.eeg_sample_rate);
    for (const auto& s : view.samples)
    {
        f.pod(s.subject_id);
        f.pod(s.stimulus);
        f.str(s.text_phrase);
        f.tensor(s.audio);
        f.tensor(s.eeg);
    }
    return f.h;
}

FeatureDiskCache::FeatureDiskCache(std::string dir, std::string revision, Mode mode)
    : dir_(std::move(dir)), revision_(std::move(revision)), mode_(mode)
{
    if (dir_.empty()) throw std::invalid_argument("FeatureDiskCache: empty directory");
    if (revision_.empty())
        throw std::invalid_argument(
            "FeatureDiskCache: empty revision; a cache without one would serve features of "
            "another code version");
    std::filesystem::create_directories(dir_);
}

auto FeatureDiskCache::from_env() -> std::optional<FeatureDiskCache>
{
    const char* dir = std::getenv("THESIS_FEATURE_CACHE");
    if (dir == nullptr || dir[0] == '\0') return std::nullopt;
    const char* rev = std::getenv("THESIS_FEATURE_CACHE_REVISION");
    const char* mode = std::getenv("THESIS_FEATURE_CACHE_MODE");
    if (rev == nullptr || rev[0] == '\0')
        throw std::invalid_argument(
            "FeatureDiskCache: THESIS_FEATURE_CACHE is set but THESIS_FEATURE_CACHE_REVISION is "
            "not; set it to the deploy's revision label (thesis_gridunesp.sbatch does)");
    const std::string m = mode == nullptr ? "" : mode;
    if (m != "read" && m != "write")
        throw std::invalid_argument(
            "FeatureDiskCache: THESIS_FEATURE_CACHE_MODE must be read or "
            "write, got \"" +
            m + "\"");
    return FeatureDiskCache(dir, rev, m == "read" ? Mode::Read : Mode::Write);
}

auto FeatureDiskCache::path_for(const std::string& key) const -> std::string
{
    Fnv f;
    f.str(revision_);
    f.str(key);
    return (std::filesystem::path(dir_) / (hex(f.h) + ".e05fc")).string();
}

auto FeatureDiskCache::load_or_compute(
    const std::string& key, std::size_t n_rows, const std::function<FeatureSet()>& compute) const
    -> FeatureSet
{
    const std::string full_key = "revision=" + revision_ + "\n" + key;
    const std::string path = path_for(key);
    auto& pm = nn::progress::ProgressManager::instance();

    std::ifstream is(path, std::ios::binary);
    if (is)
    {
        const auto corrupt = [&path](const std::string& why)
        {
            return std::runtime_error("FeatureDiskCache: " + path + " " + why +
                                      "; delete it and run the warm step again");
        };
        std::array<char, sizeof(kMagic)> magic{};
        if (!is.read(magic.data(), magic.size()) ||
            std::memcmp(magic.data(), kMagic, sizeof(kMagic)) != 0)
            throw corrupt("is not a feature-cache file of this format version");
        std::string stored_key;
        if (!read_string(is, stored_key)) throw corrupt("has an unreadable key");
        if (stored_key != full_key)
            throw corrupt("holds another key (a name collision):\n  stored:   " + stored_key +
                          "\n  expected: " + full_key);
        FeatureSet fs;
        std::uint64_t rows = 0;
        std::uint64_t cols = 0;
        if (!read_string(is, fs.label) || !read_pod(is, rows) || !read_pod(is, cols))
            throw corrupt("has an unreadable header");
        if (rows != n_rows)
            throw corrupt("has " + std::to_string(rows) + " rows for a view of " +
                          std::to_string(n_rows) + " samples");
        fs.vectors.assign(
            static_cast<std::size_t>(rows), std::vector<double>(static_cast<std::size_t>(cols)));
        for (auto& row : fs.vectors)
            if (!is.read(reinterpret_cast<char*>(row.data()),
                    static_cast<std::streamsize>(row.size() * sizeof(double))))
                throw corrupt("is truncated");
        std::uint64_t sum = 0;
        if (!read_pod(is, sum) || sum != payload_checksum(fs)) throw corrupt("fails its checksum");
        pm.log("[Thesis] feature cache: read " + fs.label + " (" + path + ")");
        return fs;
    }

    if (mode_ == Mode::Read)
        throw std::runtime_error("FeatureDiskCache: no cached features for\n" + full_key + "\n(" +
                                 path +
                                 "). The arrays read only: run the warm step first "
                                 "(thesis --warm-feature-cache, "
                                 "scripts/pipeline/thesis/thesis_warm_feature_cache.sbatch)");

    FeatureSet fs = compute();
    if (fs.vectors.size() != n_rows)
        throw std::logic_error("FeatureDiskCache: the extraction gave " +
                               std::to_string(fs.vectors.size()) + " rows for " +
                               std::to_string(n_rows) + " samples");
    const std::size_t cols = fs.vectors.empty() ? 0 : fs.vectors[0].size();
    for (const auto& row : fs.vectors)
        if (row.size() != cols)
            throw std::logic_error("FeatureDiskCache: " + fs.label + " has rows of two widths");

    // Per-writer temp name (host + pid: array tasks on different nodes share the directory),
    // then an atomic rename: a reader sees a whole file or none; identical writers race
    // harmlessly.
    const std::string tmp = path + ".tmp." + host_name() + "." + std::to_string(::getpid());
    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("FeatureDiskCache: cannot open " + tmp);
        os.write(kMagic, sizeof(kMagic));
        write_string(os, full_key);
        write_string(os, fs.label);
        write_pod(os, static_cast<std::uint64_t>(fs.vectors.size()));
        write_pod(os, static_cast<std::uint64_t>(cols));
        for (const auto& row : fs.vectors)
            os.write(reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(row.size() * sizeof(double)));
        write_pod(os, payload_checksum(fs));
        os.flush();
        if (!os) throw std::runtime_error("FeatureDiskCache: writing " + tmp + " failed");
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec)
    {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("FeatureDiskCache: cannot rename " + tmp + " to " + path);
    }
    pm.log("[Thesis] feature cache: wrote " + fs.label + " (" + path + ")");
    return fs;
}

} // namespace thesis
