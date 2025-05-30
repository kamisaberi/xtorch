#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor aria(torch::Tensor x);

    struct ARiA : xt::Module {
    public:
        ARiA() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



