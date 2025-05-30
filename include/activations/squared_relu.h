#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor squared_relu(torch::Tensor x);

    struct SquaredReLU : xt::Module {
    public:
        SquaredReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}



