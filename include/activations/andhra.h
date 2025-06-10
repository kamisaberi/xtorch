#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor andhra(torch::Tensor x, double alpha = 1.0, double beta = 0.01);

    struct ANDHRA : xt::Module {
    public:
        ANDHRA() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



