#pragma once
#include <torch/torch.h>
#include <vector>
#include <type_traits>

namespace xt
{
    class Module1 : public torch::nn::Module
    {
    public:
        // Variadic template forward method (declaration only)
        template <typename... Args>
        torch::Tensor forward(Args... args);

        // Pure virtual forward method for vector of tensors
        virtual torch::Tensor forward(std::vector<torch::Tensor> tensors) = 0;

        // Virtual destructor for proper cleanup
        virtual ~Module() = default;
    };

    // Explicit template instantiation declarations
    extern template torch::Tensor Module1::forward(torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, torch::Tensor, torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor);
    extern template torch::Tensor Module1::forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                                                   torch::Tensor, torch::Tensor);
} // namespace xt
