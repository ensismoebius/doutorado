#ifndef NN_UTILITY_PATH_EXPAND_HPP
#define NN_UTILITY_PATH_EXPAND_HPP

#include <cstdlib>
#include <string>

namespace nn::utility
{

/**
 * @file path_expand.hpp
 * @brief Expand a leading `~/` in a path to the current user's home directory.
 *
 * Profiles store the dataset root portably as `~/database.sqlite`; the sqlite
 * API and std::filesystem treat `~` literally, so any code that turns a
 * profile path into a filesystem location must expand it first.
 *
 * Only a leading `~/` is rewritten. Paths without it, or already-absolute
 * paths, are returned unchanged. If HOME is unset, `~` is left as-is so the
 * resulting error names the original string.
 */
inline auto expand_home(const std::string& path) -> std::string
{
    if (path.size() < 2 || path[0] != '~' || path[1] != '/')
    {
        return path;
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0')
    {
        return path;
    }

    return std::string(home) + path.substr(1);
}

} // namespace nn::utility

#endif // NN_UTILITY_PATH_EXPAND_HPP
