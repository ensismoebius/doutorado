/**
 * @file include/nn/io/StateIO.hpp
 * @brief Stateio.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_IO_STATEIO_HPP
#define NN_IO_STATEIO_HPP

#include <cstdint>
#include <fstream>
#include <map>
#include <string>

#include "nn/tensor/Tensor.hpp"

namespace nn::io
{
using StateDict = std::map<std::string, nn::Tensor>;
// Save a state dictionary to a binary file. Format:
// uint32_t num_entries
// for each entry:
//   uint32_t key_len, key bytes
//   uint64_t rows, uint64_t cols
//   float data[rows*cols]
// Returns true on success.
inline bool save_state_dict(const std::map<std::string, nn::Tensor>& sd, const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = static_cast<uint32_t>(sd.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (const auto& kv : sd)
    {
        const std::string& key = kv.first;
        const nn::Tensor& t = kv.second;
        uint32_t key_len = static_cast<uint32_t>(key.size());
        f.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        f.write(key.data(), key_len);
        uint64_t rows = static_cast<uint64_t>(t.rows());
        uint64_t cols = static_cast<uint64_t>(t.cols());
        f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        const float* data = t.data_ptr();
        if (rows * cols > 0)
        {
            f.write(reinterpret_cast<const char*>(data),
                static_cast<std::streamsize>(rows * cols * sizeof(float)));
        }
    }
    return f.good();
}

inline bool load_state_dict(std::map<std::string, nn::Tensor>& out, const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    for (uint32_t i = 0; i < n; ++i)
    {
        uint32_t key_len = 0;
        f.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        std::string key;
        key.resize(key_len);
        f.read(&key[0], key_len);
        uint64_t rows = 0, cols = 0;
        f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        nn::Tensor t(static_cast<nn::Index>(rows), static_cast<nn::Index>(cols));
        if (rows * cols > 0)
        {
            f.read(reinterpret_cast<char*>(t.mutable_data_ptr()),
                static_cast<std::streamsize>(rows * cols * sizeof(float)));
        }
        out.emplace(key, std::move(t));
    }
    return f.good();
}

inline StateDict load_state_dict(const std::string& path)
{
    StateDict out;
    load_state_dict(out, path);
    return out;
}

} // namespace nn::io

#endif // NN_IO_STATEIO_HPP
