#include  "include/base/module1.h"

namespace xt
{
    // Implementation of variadic forward method
    template <typename... Args>
    torch::Tensor Module1::forward(Args... args)
    {
        // Ensure all arguments are torch::Tensor
        static_assert(
            (std::is_same_v<std::decay_t<Args>, torch::Tensor> && ...),
            "All arguments to forward must be torch::Tensor"
        );

        // Collect tensors into a vector
        std::vector<torch::Tensor> tensors = {args...};

        // Check for valid number of arguments (1 to 10)
        if (tensors.empty())
        {
            throw std::invalid_argument("At least one tensor must be provided");
        }
        if (tensors.size() > 10)
        {
            throw std::invalid_argument("Maximum 10 tensors allowed");
        }

        // Call the virtual forward method
        return forward(tensors);
    }

    // Explicit template instantiations
    template torch::Tensor Module1::forward(torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                            torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                            torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                            torch::Tensor, torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                            torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                            torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
} // namespace xt
