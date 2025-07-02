#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor fem(torch::Tensor x, double alpha = 0.25, double beta = 0.01);

    struct FEM : xt::Module {
    public:
        FEM() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



