#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor rational(torch::Tensor x);

    struct Rational : xt::Module {
    public:
        Rational() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



