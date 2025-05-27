#pragma once

#include <torch/torch.h>
#include <vector>
#include <type_traits>

namespace xt {

    class Module2 : public torch::nn::Module {
    public:
        // Variadic template forward method with auto return type
        template<typename... Args>
        auto forward(Args... args) -> torch::Tensor;

        // Pure virtual destructor to make the class abstract
        virtual ~Module2() = 0;
    };

    // Pure virtual destructor definition
    inline Module2::~Module2() = default;

} // namespace xt

#include "module2.tpp"
