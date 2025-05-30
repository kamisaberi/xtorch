#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor asu(torch::Tensor x);

    struct ASU : xt::Module {
    public:
        ASU() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



