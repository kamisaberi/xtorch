#pragma once
#include <torch/torch.h>
#include <vector>
#include <type_traits>

namespace xt
{
    class Module1 : public torch::nn::Module
    {
    public:
        // Variadic template forward method
        template <typename... Args>
        torch::Tensor forward(Args... args)
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

        // Pure virtual forward method for vector of tensors
        virtual torch::Tensor forward(std::vector<torch::Tensor> tensors) = 0;

        // Virtual destructor for proper cleanup
        virtual ~Module() = default;
    };

    // Explicit template instantiations for 1 to 10 arguments
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
