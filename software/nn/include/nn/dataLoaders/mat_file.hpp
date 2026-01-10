#pragma once

#include <matio.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

/**
 * @file mat_file.hpp
 * @brief Thin RAII wrapper around `matio` for reading MATLAB .mat files.
 *
 * Ownership / lifetime:
 * - `MatFile` owns the underlying `mat_t*` handle and closes it in the destructor.
 * - `readVariable()` returns a raw `matvar_t*` allocated by matio; the caller is
 *   responsible for freeing it with `Mat_VarFree()`.
 * - `readFirstNumericVariable()` returns an owning `unique_ptr` with a custom
 *   deleter, which is safer for "read-and-map" workflows.
 */

class MatFile
{
   public:
    explicit MatFile(const std::string& filename);
    ~MatFile();

    auto getVariableNames() -> std::vector<std::string>;
    auto readVariable(const std::string& varName) -> matvar_t*;
    auto readFirstNumericVariable()
        -> std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>>;

   private:
    mat_t* matfp_ = nullptr;
};
