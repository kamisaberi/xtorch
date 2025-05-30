#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor margin_relu(torch::Tensor x);

    struct MarginReLU : xt::Module {
    public:
        MarginReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



