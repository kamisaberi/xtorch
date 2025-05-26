#pragma once
#include <torch/torch.h>
#include <vector>
#include <type_traits>

namespace xt
{
    class Module1 : public torch::nn::Module
    {
    public:
        // Variadic template forward method (non-virtual, as templates cannot be virtual)
        template <typename... Args>
        torch::Tensor forward(Args... args);

        // Pure virtual destructor to make the class abstract
        ~Module1() override = 0;
    };

    // Pure virtual destructor definition
    inline Module1::~Module1() = default;

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
