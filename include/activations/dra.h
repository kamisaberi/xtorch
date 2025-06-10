#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor dra(torch::Tensor x, double alpha = 1.0);

    struct DRA : xt::Module {
    public:
        DRA() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



