#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor smish(torch::Tensor x);

    struct Smish : xt::Module {
    public:
        Smish() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



