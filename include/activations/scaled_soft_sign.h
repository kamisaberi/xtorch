#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor scaled_soft_sign(const torch::Tensor& x, double scale_in = 1.0, double scale_out = 1.0);

    struct ScaledSoftSign : xt::Module {
    public:
        ScaledSoftSign() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



