#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor colu(torch::Tensor x, double M_val = 1.0);

    struct CoLU : xt::Module {
    public:
        CoLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



