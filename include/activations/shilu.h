#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor shilu(torch::Tensor x);

    struct ShiLU : xt::Module {
    public:
        ShiLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



