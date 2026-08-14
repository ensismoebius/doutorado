#include "GaModelSnapshot.hpp"

#include <cnpy.h>

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace pga
{

void save_state_dict_npz(
    const std::map<std::string, nn::Tensor>& state_dict, const std::string& path)
{
    if (state_dict.empty())
        throw std::invalid_argument(
            "save_state_dict_npz: state_dict is empty — refusing to write an empty snapshot to " +
            path);

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    // Write to a temp path, then atomically rename over the target — this file is
    // overwritten every time a new run-wide best is found, so a crash mid-write must
    // never leave a torn .npz sitting at the final path (see header: a permanently
    // corrupt snapshot could never self-heal if that genome stays the all-time best).
    const std::string tmp = path + ".tmp";
    bool first = true;
    for (const auto& [key, tensor] : state_dict)
    {
        cnpy::npz_save(tmp,
            key,
            tensor.data_ptr(),
            std::vector<size_t>{
                static_cast<size_t>(tensor.rows()), static_cast<size_t>(tensor.cols())},
            first ? "w" : "a");
        first = false;
    }
    std::filesystem::rename(tmp, path);
}

} // namespace pga
