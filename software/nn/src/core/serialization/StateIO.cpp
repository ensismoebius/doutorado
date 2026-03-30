#include "nn/serialization/StateIO.hpp"

#include <fstream>
#include <iostream>

namespace nn::serialization
{
bool save_state_dict(const std::map<std::string, nn::Tensor>& sd, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    // Simple binary format: magic, version, entry_count
    f.write("NNSD", 4);
    uint32_t version = 1;
    f.write(reinterpret_cast<const char*>(&version), sizeof(version));
    uint32_t entries = static_cast<uint32_t>(sd.size());
    f.write(reinterpret_cast<const char*>(&entries), sizeof(entries));
    for (const auto& kv : sd)
    {
        const std::string& key = kv.first;
        uint32_t klen = static_cast<uint32_t>(key.size());
        f.write(reinterpret_cast<const char*>(&klen), sizeof(klen));
        f.write(key.data(), klen);
        const nn::Tensor& t = kv.second;
        uint32_t rows = static_cast<uint32_t>(t.rows());
        uint32_t cols = static_cast<uint32_t>(t.cols());
        f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        const float* data = t.data_ptr();
        size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (count)
            f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    }
    return f.good();
}

std::map<std::string, nn::Tensor> load_state_dict(const std::string& path)
{
    std::map<std::string, nn::Tensor> out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    char magic[4];
    f.read(magic, 4);
    if (f.gcount() != 4) return out;
    if (std::string(magic, 4) != "NNSD") return out;
    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    uint32_t entries = 0;
    f.read(reinterpret_cast<char*>(&entries), sizeof(entries));
    for (uint32_t i = 0; i < entries; ++i)
    {
        uint32_t klen = 0;
        f.read(reinterpret_cast<char*>(&klen), sizeof(klen));
        if (!f) break;
        std::string key(klen, '\0');
        f.read(key.data(), klen);
        uint32_t rows = 0, cols = 0;
        f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        nn::Tensor t(rows, cols);
        size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (count)
        {
            f.read(reinterpret_cast<char*>(t.mutable_data_ptr()), count * sizeof(float));
        }
        out.emplace(std::move(key), std::move(t));
    }
    return out;
}

} // namespace nn::serialization
