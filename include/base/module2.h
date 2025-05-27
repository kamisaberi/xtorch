#pragma once

#include <torch/torch.h>
#include <vector>
#include <type_traits>

namespace xt {

    class Module : public torch::nn::Module {
    public:
        // Variadic template forward method with auto return type
        template<typename... Args>
        auto forward(Args... args) -> torch::Tensor;

        // Pure virtual destructor to make the class abstract
        virtual ~Module() = 0;
    };

    // Pure virtual destructor definition
    inline Module::~Module() = default;

} // namespace xt

#include "xt_module.tpp"
