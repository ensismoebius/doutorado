/**
 * @file include/nn/testing/tempfile.hpp
 * @brief Tempfile.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_TESTING_TEMPFILE_HPP
#define NN_TESTING_TEMPFILE_HPP

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace nn::testing
{
// Create a unique temporary file in the system temp directory and return its path.
// The file is created (empty) and closed. Caller may remove it when done.
inline std::string make_temp_file(const std::string& prefix = "nn_test_")
{
    // Template for mkstemp. Use /tmp for portability on POSIX.
    std::string tmpl = std::string("/tmp/") + prefix + "XXXXXX";
    // mkstemp requires a modifiable char*.
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = mkstemp(buf.data());
    if (fd == -1)
    {
        return std::string();
    }
    // Close descriptor; caller will open if needed.
    close(fd);
    return std::string(buf.data());
}

// RAII TempFile: creates a temp file and removes it on destruction unless released.
class TempFile
{
   public:
    explicit TempFile(const std::string& prefix = "nn_test_")
    {
        path_ = make_temp_file(prefix);
        owned_ = !path_.empty();
    }
    ~TempFile()
    {
        if (owned_ && !path_.empty())
        {
            std::remove(path_.c_str());
        }
    }

    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    TempFile(TempFile&& o) noexcept : path_(std::move(o.path_)), owned_(o.owned_)
    {
        o.owned_ = false;
    }

    TempFile& operator=(TempFile&& o) noexcept
    {
        if (this != &o)
        {
            if (owned_ && !path_.empty()) std::remove(path_.c_str());
            path_ = std::move(o.path_);
            owned_ = o.owned_;
            o.owned_ = false;
        }
        return *this;
    }

    const std::string& path() const
    {
        return path_;
    }

    // Release ownership; file will not be removed on destruction.
    std::string release()
    {
        owned_ = false;
        return std::move(path_);
    }

    // Convenience write
    bool write(const std::string& data, std::ios::openmode mode = std::ios::out)
    {
        if (path_.empty()) return false;
        std::ofstream ofs(path_, mode | std::ios::binary);
        if (!ofs) return false;
        ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
        return ofs.good();
    }

   private:
    std::string path_;
    bool owned_ = false;
};

} // namespace nn::testing

#endif // NN_TESTING_TEMPFILE_HPP
