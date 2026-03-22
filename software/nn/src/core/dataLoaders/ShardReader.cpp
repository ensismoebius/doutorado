#include "nn/dataLoaders/ShardReader.hpp"

#include <cnpy.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace nn::dataLoaders
{
std::unordered_map<std::string, void*> ShardReader::open_shards_{};

auto ShardReader::read_file_to_string(const std::string& path) -> std::string
{
    std::ifstream in(path);
    if (!in)
    {
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

ShardReader::ShardReader(const std::string& index_path) : index_path_(index_path)
{
    load_index();
}

ShardReader::~ShardReader() = default;

void ShardReader::load_index()
{
    index_.audio.clear();
    index_.eeg.clear();

    const std::string s = read_file_to_string(index_path_);
    if (s.empty())
    {
        return;
    }

    auto parse_section = [&](const std::string& section_name, std::vector<ShardEntry>& out)
    {
        std::size_t pos = 0;
        const std::string needle = '"' + section_name + '"';
        pos = s.find(needle);
        if (pos == std::string::npos)
        {
            return;
        }

        pos = s.find('[', pos);
        if (pos == std::string::npos)
        {
            return;
        }

        const std::size_t end = s.find(']', pos);
        if (end == std::string::npos)
        {
            return;
        }

        const std::string body = s.substr(pos + 1, end - pos - 1);
        std::size_t cur = 0;
        while (true)
        {
            const std::size_t obj_start = body.find('{', cur);
            if (obj_start == std::string::npos)
            {
                break;
            }

            const std::size_t obj_end = body.find('}', obj_start);
            if (obj_end == std::string::npos)
            {
                break;
            }

            const std::string obj = body.substr(obj_start + 1, obj_end - obj_start - 1);
            ShardEntry e;

            const auto fpos = obj.find("\"file\"");
            if (fpos != std::string::npos)
            {
                const auto colon = obj.find(':', fpos);
                if (colon != std::string::npos)
                {
                    const auto q1 = obj.find('"', colon);
                    const auto q2 = obj.find('"', q1 + 1);
                    if (q1 != std::string::npos && q2 != std::string::npos)
                    {
                        e.file = obj.substr(q1 + 1, q2 - q1 - 1);
                    }
                }
            }

            const auto spos = obj.find("\"start\"");
            if (spos != std::string::npos)
            {
                const auto colon = obj.find(':', spos);
                if (colon != std::string::npos)
                {
                    const auto comma = obj.find(',', colon);
                    const auto num = obj.substr(
                        colon + 1, (comma == std::string::npos ? obj.size() : comma) - colon - 1);
                    e.start = static_cast<std::size_t>(std::stoul(num));
                }
            }

            const auto cpos = obj.find("\"count\"");
            if (cpos != std::string::npos)
            {
                const auto colon = obj.find(':', cpos);
                if (colon != std::string::npos)
                {
                    const auto comma = obj.find(',', colon);
                    const auto num = obj.substr(
                        colon + 1, (comma == std::string::npos ? obj.size() : comma) - colon - 1);
                    e.count = static_cast<std::size_t>(std::stoul(num));
                }
            }

            out.push_back(e);
            cur = obj_end + 1;
        }
    };

    parse_section("audio", index_.audio);
    parse_section("eeg", index_.eeg);
}

static auto read_row_from_npz(const std::string& fullpath, const std::string& var_name, std::size_t offset)
    -> std::optional<std::vector<double>>
{
    try
    {
        auto npz = cnpy::npz_load(fullpath);
        const auto it = npz.find(var_name);
        if (it == npz.end())
        {
            return std::nullopt;
        }

        const auto arr = it->second;
        if (arr.shape.size() != 2)
        {
            return std::nullopt;
        }

        const std::size_t cols = arr.shape[1];
        std::vector<double> out(cols);
        if (arr.word_size == sizeof(double))
        {
            const double* data = arr.data<double>();
            for (std::size_t c = 0; c < cols; ++c)
            {
                out[c] = data[offset * cols + c];
            }
        }
        else if (arr.word_size == sizeof(float))
        {
            const float* data = arr.data<float>();
            for (std::size_t c = 0; c < cols; ++c)
            {
                out[c] = static_cast<double>(data[offset * cols + c]);
            }
        }
        else
        {
            return std::nullopt;
        }

        return out;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

auto ShardReader::locate_audio_row(std::size_t row) const -> std::optional<Loc>
{
    std::size_t r = row;
    for (const auto& e : index_.audio)
    {
        if (r < e.count)
        {
            return Loc{std::string(e.file), r};
        }
        r -= e.count;
    }
    return std::nullopt;
}

auto ShardReader::locate_eeg_row(std::size_t row) const -> std::optional<Loc>
{
    std::size_t r = row;
    for (const auto& e : index_.eeg)
    {
        if (r < e.count)
        {
            return Loc{std::string(e.file), r};
        }
        r -= e.count;
    }
    return std::nullopt;
}

auto ShardReader::read_row_from_shard(
    const std::string& shard_fullpath, const std::string& var_name, std::size_t offset)
    -> std::optional<std::vector<double>>
{
    const auto it = ShardReader::open_shards_.find(shard_fullpath);
    if (it != ShardReader::open_shards_.end())
    {
        try
        {
            auto* arrmap = static_cast<cnpy::npz_t*>(it->second);
            if (!arrmap)
            {
                return std::nullopt;
            }

            const auto ait = arrmap->find(var_name);
            if (ait == arrmap->end())
            {
                return std::nullopt;
            }

            const auto arr = ait->second;
            if (arr.shape.size() != 2)
            {
                return std::nullopt;
            }

            const std::size_t cols = arr.shape[1];
            std::vector<double> out(cols);
            if (arr.word_size == sizeof(double))
            {
                const double* data = arr.data<double>();
                for (std::size_t c = 0; c < cols; ++c)
                {
                    out[c] = data[offset * cols + c];
                }
            }
            else if (arr.word_size == sizeof(float))
            {
                const float* data = arr.data<float>();
                for (std::size_t c = 0; c < cols; ++c)
                {
                    out[c] = static_cast<double>(data[offset * cols + c]);
                }
            }
            else
            {
                return std::nullopt;
            }

            return out;
        }
        catch (...)
        {
            return std::nullopt;
        }
    }

    return read_row_from_npz(shard_fullpath, var_name, offset);
}

void ShardReader::preopen_shard(const std::string& shard_fullpath)
{
    try
    {
        auto npz = cnpy::npz_load(shard_fullpath);
        auto* ptr = new cnpy::npz_t(std::move(npz));
        ShardReader::open_shards_.emplace(shard_fullpath, static_cast<void*>(ptr));
    }
    catch (...)
    {
        // Ignore preopen errors and keep on-demand loading path.
    }
}

void ShardReader::preopen_index_shards(const std::string& index_parent)
{
    for (const auto& e : index_.audio)
    {
        const std::filesystem::path p = std::filesystem::path(index_parent) / e.file;
        preopen_shard(p.string());
    }
    for (const auto& e : index_.eeg)
    {
        const std::filesystem::path p = std::filesystem::path(index_parent) / e.file;
        preopen_shard(p.string());
    }
}
} // namespace nn::dataLoaders
