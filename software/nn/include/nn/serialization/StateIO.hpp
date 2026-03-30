// Single clean header-only implementation
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "nn/tensor/Tensor.hpp"

namespace nn::serialization
{
using StateDict = std::map<std::string, nn::Tensor>;

inline bool save_state_dict(const StateDict& sd, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t n = static_cast<uint64_t>(sd.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& kv : sd)
    {
        const std::string& name = kv.first;
        const nn::Tensor& t = kv.second;
        uint64_t name_len = static_cast<uint64_t>(name.size());
        f.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        if (name_len) f.write(name.data(), static_cast<std::streamsize>(name_len));
        uint64_t rows = static_cast<uint64_t>(t.rows());
        uint64_t cols = static_cast<uint64_t>(t.cols());
        f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        uint64_t count = rows * cols;
        if (count > 0)
        {
            const float* data = t.data_ptr();
            f.write(reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(count * sizeof(float)));
        }
    }
    return f.good();
}

inline StateDict load_state_dict(const std::string& path)
{
    StateDict out;
    std::ifstream f(path, std::ios::binary);
    if (!f) return out;
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    for (uint64_t i = 0; i < n; ++i)
    {
        uint64_t name_len = 0;
        f.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name;
        name.resize(static_cast<size_t>(name_len));
        if (name_len > 0) f.read(&name[0], static_cast<std::streamsize>(name_len));
        uint64_t rows = 0, cols = 0;
        f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        nn::Tensor t(static_cast<nn::Index>(rows), static_cast<nn::Index>(cols));
        uint64_t count = rows * cols;
        if (count > 0)
        {
            f.read(reinterpret_cast<char*>(t.mutable_data_ptr()),
                static_cast<std::streamsize>(count * sizeof(float)));
        }
        out.emplace(std::move(name), std::move(t));
    }
    return out;
}

} // namespace nn::serialization
