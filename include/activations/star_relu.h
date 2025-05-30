#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor star_relu(torch::Tensor x);

    struct StarReLU: xt::Module {
    public:
        StarReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



