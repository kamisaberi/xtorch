#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor scaled_soft_sign(torch::Tensor x);

    struct ScaledSoftSign : xt::Module {
    public:
        ScaledSoftSign() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



