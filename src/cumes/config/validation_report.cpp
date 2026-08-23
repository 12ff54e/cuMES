// validation_report.cpp — collected validation findings.
#include "cumes/config/validation_report.hpp"

namespace cumes {

void ValidationReport::error(std::string key, std::string message) {
    issues_.push_back(
        ValidationIssue{Severity::ERROR, std::move(key), std::move(message)});
}

void ValidationReport::warn(std::string key, std::string message) {
    issues_.push_back(
        ValidationIssue{Severity::WARNING, std::move(key), std::move(message)});
}

bool ValidationReport::has_errors() const {
    for (const auto& issue : issues_) {
        if (issue.severity == Severity::ERROR) return true;
    }
    return false;
}

std::vector<std::string> ValidationReport::errors() const {
    std::vector<std::string> out;
    for (const auto& issue : issues_) {
        if (issue.severity == Severity::ERROR) out.push_back(issue.message);
    }
    return out;
}

std::vector<std::string> ValidationReport::warnings() const {
    std::vector<std::string> out;
    for (const auto& issue : issues_) {
        if (issue.severity == Severity::WARNING) out.push_back(issue.message);
    }
    return out;
}

}  // namespace cumes
