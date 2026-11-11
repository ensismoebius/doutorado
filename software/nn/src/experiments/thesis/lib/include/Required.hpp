#pragma once

/**
 * @file Required.hpp
 * @brief A configuration value with no default: it holds what the profile declared, or nothing.
 *
 * WHY THIS EXISTS. A struct member written `int epochs = 50;` answers every read, declared or
 * not. A profile that forgets `training.epochs` therefore trains for 50 epochs, the summary
 * records 50, and nothing anywhere says that 50 was never chosen. The thesis profiles hit the
 * worst case of this: `autoencoder.model` defaulted to "lstm-ae", the extractor the thesis
 * dropped, so a profile without `model` silently trained the wrong network.
 *
 * `Required<int> epochs{"training.epochs"}` has no value until something assigns one. The
 * string is the field's NAME (its profile path), used in error messages; it is not a value.
 * Reading an undeclared field throws, naming it:
 *
 *     profile declares          read of cfg.training.epochs
 *     ------------------------  -----------------------------------------------------------
 *     "epochs": 60              60
 *     (nothing)                 throws: "training.epochs" is read but was never declared
 *
 * The throw is the backstop, not the main guard: ThesisConfig's parser already refuses a
 * profile that omits a key the run will read. What `Required` adds is that code building a
 * config by hand (tests, bindings, the GA) gets the same guarantee: nothing can read a value
 * that nobody chose.
 */

#include <concepts>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "nlohmann/json.hpp"

namespace thesis
{

template <typename T>
class Required
{
   public:
    using value_type = T;

    /// `path` is the field's profile path ("training.epochs") — a label for error messages.
    explicit Required(const char* path) : path_(path) {}

    /// Declare the value.
    auto operator=(T value) -> Required&
    {
        value_ = std::move(value);
        return *this;
    }

    [[nodiscard]] auto declared() const noexcept -> bool
    {
        return value_.has_value();
    }
    [[nodiscard]] auto path() const noexcept -> const char*
    {
        return path_;
    }

    /// The declared value. Throws if nothing was declared — there is no default to return.
    [[nodiscard]] auto get() const -> const T&
    {
        if (!value_)
            throw std::logic_error(std::string("Config value \"") + path_ +
                                   "\" is read but was never declared. It has no default: "
                                   "declare it in the profile (or, for a config built in code, "
                                   "assign it before use).");
        return *value_;
    }

    // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions): reads as T.
    operator const T&() const
    {
        return get();
    }
    auto operator*() const -> const T&
    {
        return get();
    }
    auto operator->() const -> const T*
    {
        return &get();
    }

    /// `cfg.model == "snn-ae"`: std::string's == is a template, which never sees through a
    /// conversion operator, so comparison is spelled out once here.
    template <typename U>
        requires(!std::is_same_v<std::remove_cvref_t<U>, Required>) &&
                requires(const T& t, const U& u) {
                    { t == u } -> std::convertible_to<bool>;
                }
    friend auto operator==(const Required& r, const U& u) -> bool
    {
        return r.get() == u;
    }

    friend auto operator<<(std::ostream& os, const Required& r) -> std::ostream&
    {
        return os << r.get();
    }

    friend auto operator+(const std::string& a, const Required& r) -> std::string
        requires std::is_same_v<T, std::string>
    {
        return a + r.get();
    }
    friend auto operator+(const Required& r, const std::string& b) -> std::string
        requires std::is_same_v<T, std::string>
    {
        return r.get() + b;
    }

    /// Range-for over a declared container (`for (const auto& s : cfg.layer_spec)`).
    friend auto begin(const Required& r)
        requires requires(const T& t) { t.begin(); }
    {
        return r.get().begin();
    }
    friend auto end(const Required& r)
        requires requires(const T& t) { t.end(); }
    {
        return r.get().end();
    }

   private:
    const char* path_;
    std::optional<T> value_;
};

/// Serializing an undeclared field throws like any other read.
template <typename T>
void to_json(nlohmann::json& j, const Required<T>& r)
{
    j = r.get();
}

} // namespace thesis
