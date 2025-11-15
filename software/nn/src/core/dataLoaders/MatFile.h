#pragma once

#include <matio.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

class MatFile
{
   public:
    MatFile(const std::string& filename);
    ~MatFile();

    auto getVariableNames() -> std::vector<std::string>;
    auto readVariable(const std::string& varName) -> matvar_t*;
    auto readFirstNumericVariable()
        -> std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>>;

   private:
    mat_t* matfp_ = nullptr;
};
