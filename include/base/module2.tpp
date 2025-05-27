#pragma once

#include "module2.h"

namespace xt {

    template<typename... Args>
    auto Module2::forward(Args... args) -> torch::Tensor {
        // Ensure all arguments are torch::Tensor
        static_assert(
            (std::is_same_v<std::decay_t<Args>, torch::Tensor> && ...),
            "All arguments to forward must be torch::Tensor"
        );

        // Collect tensors into a vector
        std::vector<torch::Tensor> tensors = {args...};

        // Check for valid number of arguments (1 to 10)
        if (tensors.empty()) {
            throw std::invalid_argument("At least one tensor must be provided");
        }
        if (tensors.size() > 10) {
            throw std::invalid_argument("Maximum 10 tensors allowed");
        }

        // Check that all tensors have the same shape
        auto reference_sizes = tensors[0].sizes();
        for (size_t i = 1; i < tensors.size(); ++i) {
            if (tensors[i].sizes() != reference_sizes) {
                throw std::invalid_argument("All tensors must have the same shape");
            }
        }

        // Sum all tensors (default implementation)
        torch::Tensor result = tensors[0];
        for (size_t i = 1; i < tensors.size(); ++i) {
            result = result + tensors[i];
        }
        return result;
    }

} // namespace xt

