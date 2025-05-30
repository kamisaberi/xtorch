#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor tanh_exp(torch::Tensor x);

    struct TanhExp : xt::Module {
    public:
        TanhExp() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



