#ifndef CUMES_CORE_ERROR_HPP_
#define CUMES_CORE_ERROR_HPP_

#include <stdexcept>

namespace cumes {

class CumesError : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};

}  // namespace cumes

#endif  // CUMES_CORE_ERROR_HPP_
