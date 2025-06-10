#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor asu(torch::Tensor x, double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

    struct ASU : xt::Module {
    public:
        ASU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



