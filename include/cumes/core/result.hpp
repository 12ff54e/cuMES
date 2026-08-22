// result.hpp — minimal value-or-error result type.
//
// Replaces exceptions at the module boundaries of the validated host model
// (config, I/O). The default error type is std::string (a single message);
// validate() uses a richer ValidationReport as its error. Phase 3 extends the
// same type to the CUDA/cuFFT status conversion at the runtime boundary.
//
// A Result is either a value (has_value() == true) or an error. The two
// optional slots are mutually exclusive by construction (the constructors are
// the only way in). This is deliberately tiny — no monadic bind/and_then yet;
// those arrive with the first consumer that needs them.
#pragma once

#include <optional>
#include <string>
#include <utility>

namespace cumes {

template <class T, class E = std::string>
class BasicResult {
   public:
    // Value constructors.
    BasicResult(const T& value) : value_(value) {}
    BasicResult(T&& value) : value_(std::move(value)) {}
    // Error constructors.
    BasicResult(const E& error) : error_(error) {}
    BasicResult(E&& error) : error_(std::move(error)) {}

    bool has_value() const { return !error_.has_value(); }
    explicit operator bool() const { return has_value(); }

    const T& value() const { return *value_; }
    T& value() { return *value_; }
    const E& error() const { return *error_; }

    // Error-construction shorthand: `return Err<T>("...")` / `Err("...")` in a
    // function returning Result<T>. Kept out-of-line-ish via CTAD below.
   private:
    std::optional<T> value_;
    std::optional<E> error_;
};

// void value specialization (the error still carries a message).
template <class E>
class BasicResult<void, E> {
   public:
    BasicResult() = default;
    BasicResult(const E& error) : error_(error) {}
    BasicResult(E&& error) : error_(std::move(error)) {}

    bool has_value() const { return !error_.has_value(); }
    explicit operator bool() const { return has_value(); }
    void value() const {}
    const E& error() const { return *error_; }

   private:
    std::optional<E> error_;
};

// Convenience aliases.
template <class T>
using Result = BasicResult<T, std::string>;
using Status = BasicResult<void, std::string>;

}  // namespace cumes
