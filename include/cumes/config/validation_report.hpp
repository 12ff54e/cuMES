// validation_report.hpp — collected validation findings (blueprint §6.1).
//
// Unlike the legacy parser (which threw on the first error), the validated
// model collects every error and warning so a single pass reports the whole
// picture. An error makes the problem invalid; a warning preserves the parse
// but is recorded (e.g. a skipped out-of-range boundary harmonic, or an
// unknown-key in compatibility mode).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cumes {

enum class Severity : std::uint8_t { kError, kWarning };

struct ValidationIssue {
    Severity severity = Severity::kError;
    std::string key;      // the offending input key, or empty
    std::string message;  // human-readable, matches the legacy fragments
};

class ValidationReport {
 public:
    void error(std::string key, std::string message);
    void warn(std::string key, std::string message);

    bool ok() const { return !has_errors(); }
    bool has_errors() const;

    const std::vector<ValidationIssue>& issues() const { return issues_; }
    std::vector<std::string> errors() const;    // messages of severity Error
    std::vector<std::string> warnings() const;  // messages of severity Warning

 private:
    std::vector<ValidationIssue> issues_;
};

}  // namespace cumes
