#pragma once

#include <matio.h>

#include <memory>
#include <optional>
#include <string>

namespace nn::dataLoaders
{

class IMatLoader
{
   public:
    virtual ~IMatLoader() = default;

    // Open a MAT file. Returns true on success, false otherwise.
    virtual bool open(const std::string& filePath) noexcept = 0;

    // Close the currently opened MAT file.
    virtual void close() noexcept = 0;

    // Read a named variable from the open MAT file. Returns nullptr if not found.
    virtual std::unique_ptr<matvar_t, void (*)(matvar_t*)> readVariable(
        const std::string& name) = 0;

    // Read the first numeric variable (convenience helper). Returns empty optional on failure.
    virtual std::optional<std::unique_ptr<matvar_t, void (*)(matvar_t*)>>
    readFirstNumericVariable() = 0;

    // Return the file path currently opened, or empty if none.
    virtual std::string filePath() const noexcept = 0;
};

} // namespace nn::dataLoaders
