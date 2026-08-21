#pragma once
// cumes_test_format.h — std::format-compatible formatting for tests.
//
// std::format lives in <format>, which libstdc++ only provides from GCC 13.
// The pinned CUDA host compiler is g++-12 (CUDA 12.1 supports gcc <= 12), so
// the .cu tests' host pass cannot use std::format. This header exposes
// cumes::test::format(...) with std::format's {} syntax: it IS std::format
// when the toolchain provides it, and falls back to a small ostream-based
// implementation otherwise. When every compiler in the build reaches gcc 13,
// the fallback disappears and only std::format remains.
//
// Only the format specifiers the tests actually use are supported: {} for
// any streamable value, {:.Nf} / {:.Ne} / {:.Ng} for floating point
// (printf %f / %e / %g semantics, precision N, default 6), and {{ / }} for
// literal braces.
#if __has_include(<format>)

#include <format>

namespace cumes::test {
using std::format;
}

#else  // g++-12 fallback (no <format> in libstdc++-12)

#include <charconv>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace cumes::test {
namespace detail {

// One printf-style floating-point field: kind selects fixed/scientific/
// default notation. Writes with saved/restored stream state so a formatted
// number never leaks flags or precision into later {} fields.
struct format_number {
    long double v;
    int prec;
    char kind;  // 'f' | 'e' | 'g'
};
inline std::ostream& operator<<(std::ostream& os, const format_number& f) {
    const auto flags = os.flags();
    const auto prec = os.precision();
    if (f.kind == 'e') {
        os << std::scientific;
    } else if (f.kind == 'f') {
        os << std::fixed;
    } else {
        os << std::defaultfloat;
    }
    os << std::setprecision(f.prec) << f.v;
    os.flags(flags);
    os.precision(prec);
    return os;
}

// spec is the field body between '{' and '}': "" or ":<digits><f|e|g>".
template <typename T>
void append_arg(std::ostringstream& os, std::string_view spec, const T& v) {
    if constexpr (std::is_floating_point_v<T>) {
        if (spec.size() >= 2 && spec[0] == ':') {
            const char kind = spec.back();
            int prec = 6;
            if (spec.size() > 2 && spec[1] == '.') {
                const char* first = spec.data() + 2;
                const char* last = spec.data() + spec.size() - 1;
                std::from_chars(first, last, prec);
            }
            if (kind == 'f' || kind == 'e' || kind == 'g') {
                os << format_number{static_cast<long double>(v), prec, kind};
                return;
            }
        }
    }
    os << v;
}

// Append literal text, collapsing '}}' into '}' (std::format escape).
inline void append_literal(std::ostringstream& os, std::string_view lit) {
    std::size_t d = 0;
    while ((d = lit.find("}}", d)) != std::string_view::npos) {
        os << lit.substr(0, d) << '}';
        lit = lit.substr(d + 2);
        d = 0;
    }
    os << lit;
}

}  // namespace detail

template <typename... Args>
std::string format(std::string_view fmt, const Args&... args) {
    std::ostringstream os;
    os << std::setprecision(6);  // printf %g default
    auto t = std::tuple<const Args&...>(args...);
    std::size_t idx = 0;
    std::size_t pos = 0;
    while (pos < fmt.size()) {
        const auto open = fmt.find('{', pos);
        if (open == std::string_view::npos) {
            detail::append_literal(os, fmt.substr(pos));
            break;
        }
        detail::append_literal(os, fmt.substr(pos, open - pos));
        // '{{' escapes a literal brace (std::format convention).
        if (open + 1 < fmt.size() && fmt[open + 1] == '{') {
            os << '{';
            pos = open + 2;
            continue;
        }
        const auto close = fmt.find('}', open + 1);
        if (close == std::string_view::npos) {  // stray '{' is literal
            detail::append_literal(os, fmt.substr(open));
            break;
        }
        const std::string_view spec = fmt.substr(open + 1, close - open - 1);
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            (void)((idx == I ? (detail::append_arg(os, spec, std::get<I>(t)), 0) : 0), ...);
        }(std::index_sequence_for<Args...>{});
        ++idx;
        pos = close + 1;
    }
    return os.str();
}

}  // namespace cumes::test

#endif
