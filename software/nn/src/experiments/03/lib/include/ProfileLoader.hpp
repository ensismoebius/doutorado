#pragma once

#include <optional>
#include <string>

#include "cli.hpp"

namespace experiment03
{
// Load a profile by name (without extension). Searches common locations.
// Returns true on success and fills out_config; on failure returns false and
// sets out_error with a diagnostic message.
auto load_profile_to_config(
    const std::string& profile_name, Config& out_config, std::string& out_error) -> bool;
} // namespace experiment03
