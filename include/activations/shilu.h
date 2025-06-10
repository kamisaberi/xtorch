#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor shilu(const torch::Tensor& x, double a = 1.0, double b = 0.0);

    struct ShiLU : xt::Module {
    public:
        ShiLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



