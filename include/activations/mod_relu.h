#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor mod_relu(torch::Tensor x);

    struct ModReLU : xt::Module {
    public:
        ModReLU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



