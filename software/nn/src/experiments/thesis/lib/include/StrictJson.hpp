#pragma once

/**
 * @file StrictJson.hpp
 * @brief Exact JSON readers for profiles whose fields are `Required<T>` (ThesisConfig, GaConfig).
 *
 * nlohmann's get<int>() on 1.5 returns 1 and get<std::uint32_t>() on -1 returns 4294967295:
 * a typo in a profile would quietly become a different number. And a key no parser asks for
 * (a misspelling, a removed option) used to be skipped without a word. These helpers read a
 * value only if its JSON type is exactly the field's type, and refuse keys nobody reads.
 */

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "nlohmann/json.hpp"

namespace thesis::strict
{

inline auto describe(const nlohmann::json& v) -> std::string
{
    return std::string(v.type_name()) + " " + v.dump();
}

[[noreturn]] inline void wrong_type(
    const std::string& owner, const char* path, const char* expected, const nlohmann::json& v)
{
    throw std::invalid_argument(
        owner + ": \"" + path + "\" must be " + expected + ", got " + describe(v) + ".");
}

/// The value at `v` as T, refusing any JSON type that is not exactly T's.
template <typename T>
auto read_exact(const nlohmann::json& v, const std::string& owner, const char* path) -> T
{
    if constexpr (std::is_same_v<T, bool>)
    {
        if (!v.is_boolean()) wrong_type(owner, path, "true or false", v);
        return v.get<bool>();
    }
    else if constexpr (std::is_integral_v<T>)
    {
        if (!v.is_number_integer()) wrong_type(owner, path, "an integer", v);
        if (v.is_number_unsigned())
        {
            if (v.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
                wrong_type(owner, path, "an integer in range", v);
        }
        else
        {
            const auto s = v.get<std::int64_t>();
            if (s < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
                (s > 0 && static_cast<std::uint64_t>(s) >
                              static_cast<std::uint64_t>(std::numeric_limits<T>::max())))
                wrong_type(owner, path, "an integer in range", v);
        }
        return v.get<T>();
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        if (!v.is_number()) wrong_type(owner, path, "a number", v);
        return v.get<T>();
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        if (!v.is_string()) wrong_type(owner, path, "a string", v);
        return v.get<std::string>();
    }
    else
    {
        using E = typename T::value_type;
        static_assert(std::is_same_v<T, std::vector<E>>, "unsupported field type");
        if (!v.is_array()) wrong_type(owner, path, "an array", v);
        T out;
        for (const auto& item : v) out.push_back(read_exact<E>(item, owner, path));
        return out;
    }
}

/// The node at dotted `path` ("a.b.c"), or nullptr when any step is absent.
inline auto find_path(const nlohmann::json& root, std::string_view path) -> const nlohmann::json*
{
    const nlohmann::json* node = &root;
    std::size_t start = 0;
    while (true)
    {
        const std::size_t dot = path.find('.', start);
        const std::string key(path.substr(start, dot - start));
        if (!node->is_object() || !node->contains(key)) return nullptr;
        node = &(*node)[key];
        if (dot == std::string_view::npos) return node;
        start = dot + 1;
    }
}

/// Every proper prefix of every field path ("a.b.c" -> "a", "a.b"): the objects a profile
/// may contain.
inline auto sections_of(const std::set<std::string>& fields) -> std::set<std::string>
{
    std::set<std::string> sections;
    for (const auto& path : fields)
        for (auto dot = path.find('.'); dot != std::string::npos; dot = path.find('.', dot + 1))
            sections.insert(path.substr(0, dot));
    return sections;
}

/// Refuse every key of `node` (recursively) that is neither a field nor a section. Top-level
/// keys listed in `skip` belong to another parser and are not visited.
inline void refuse_unknown_keys(const nlohmann::json& node,
    const std::string& prefix,
    const std::set<std::string>& fields,
    const std::set<std::string>& sections,
    const std::vector<std::string>& skip,
    const std::string& owner)
{
    if (!node.is_object())
        throw std::invalid_argument(owner + ": \"" + (prefix.empty() ? "<root>" : prefix) +
                                    "\" must be a JSON object, got " + describe(node) + ".");
    for (const auto& [key, value] : node.items())
    {
        if (prefix.empty() && std::find(skip.begin(), skip.end(), key) != skip.end()) continue;
        const std::string path = prefix.empty() ? key : prefix + "." + key;
        if (fields.count(path) != 0) continue;
        if (sections.count(path) != 0)
        {
            refuse_unknown_keys(value, path, fields, sections, skip, owner);
            continue;
        }
        throw std::invalid_argument(owner + ": unknown key \"" + path +
                                    "\" in the profile — no parameter has this name. Fix the "
                                    "name or remove the key.");
    }
}

/// The declaration rule for one field: declared exactly when the program reads it.
/// `when` is the reading condition in words ("always" for an unconditional field).
template <typename Field>
void check_declared(const Field& field, bool reads, const char* when, const std::string& owner)
{
    const std::string path = field.path();
    const bool unconditional = std::string_view(when) == "always";
    if (reads && !field.declared())
        throw std::invalid_argument(
            owner + ": the profile does not declare \"" + path + "\", which this run reads (" +
            (unconditional ? std::string("every run does") : std::string(when)) +
            "). There is no default: declare it.");
    if (!reads && field.declared())
        throw std::invalid_argument(owner + ": the profile declares \"" + path +
                                    "\", which this run never reads (it is read only when " + when +
                                    "). Remove it: a declared value that changes nothing reads "
                                    "as a choice that was never applied.");
}

} // namespace thesis::strict
