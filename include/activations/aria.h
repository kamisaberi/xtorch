#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor aria(torch::Tensor x, double alpha = 1.0, double beta = 1.0);

    struct ARiA : xt::Module {
    public:
        ARiA() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



